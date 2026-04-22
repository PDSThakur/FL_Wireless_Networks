"""
run_fedavg.py — Main experiment runner for FedAvg baseline

Usage:
    python run_fedavg.py --dataset cifar10 --num_clients 10 --num_rounds 100 --alpha 0.5
    python run_fedavg.py --dataset fmnist  --num_clients 50 --num_rounds 150 --alpha iid
    python run_fedavg.py --run_all   # Run all required configurations
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import flwr as fl
from flwr.simulation import run_simulation

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model  import get_model
from data   import (
    load_full_dataset,
    get_partition,
    get_client_dataloader,
    get_test_dataloader,
    get_backdoor_test_loaders,
    partition_stats,
)
from utils  import MetricsTracker, compute_comm_cost_mb, plot_accuracy_vs_rounds, plot_loss_vs_rounds, plot_iid_vs_noniid, print_results_table
from client import build_client_app
from server import build_server_app


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def resolve_devices(client_num_gpus: float) -> tuple[torch.device, torch.device]:
    """Resolve server/client devices based on CUDA availability and Ray resources."""
    has_cuda = torch.cuda.is_available()
    server_device = torch.device("cuda" if has_cuda else "cpu")
    client_device = torch.device("cuda" if has_cuda and client_num_gpus > 0.0 else "cpu")

    if client_num_gpus > 0.0 and not has_cuda:
        print("  [Device] Clients requested GPU resources, but CUDA is unavailable. Using CPU.")
    elif client_num_gpus == 0.0 and has_cuda:
        print("  [Device] Ray clients configured with num_gpus=0.0. Using CPU for clients.")

    return server_device, client_device


# ─────────────────────────────────────────────
# Single Experiment
# ─────────────────────────────────────────────
def run_experiment(
    dataset: str,
    num_clients: int,
    num_rounds: int,
    alpha,                      # float or "iid"
    local_epochs: int   = 5,
    batch_size: int     = 32,
    learning_rate: float = 0.01,
    momentum: float     = 0.9,
    fraction_fit: float = 0.5,
    client_num_gpus: float = 0.0,
    backdoor_target_label: int = 0,
    semantic_source_label: int = 1,
    pixel_trigger_size: int = 3,
    pixel_trigger_value: float = 1.0,
    disable_backdoor_eval: bool = False,
    seed: int           = 42,
    results_dir: str    = "../results",
) -> dict:
    """
    Run a single FedAvg experiment with the given configuration.
    Returns a summary dict of final metrics.
    """
    set_seeds(seed)
    server_device, client_device = resolve_devices(client_num_gpus)

    alpha_str = str(alpha)
    exp_name  = f"fedavg_{dataset}_c{num_clients}_r{num_rounds}_a{alpha_str}"

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  Dataset:    {dataset.upper()}")
    print(f"  Clients:    {num_clients}  |  Rounds: {num_rounds}")
    print(f"  Alpha:      {alpha_str}")
    print(f"  Server:     {server_device}  |  Client: {client_device} (Ray num_gpus={client_num_gpus})")
    print(f"{'='*60}")

    # ── Data Preparation ──────────────────────
    print("\n[1/4] Loading and partitioning data...")
    train_dataset, test_dataset = load_full_dataset(dataset)
    client_indices = get_partition(train_dataset, num_clients, alpha, seed)
    partition_stats(train_dataset, client_indices)

    # Split each client's data: 90% train, 10% validation
    train_loaders, val_loaders = [], []
    for indices in client_indices:
        n_val   = max(1, int(0.1 * len(indices)))
        val_idx = indices[:n_val]
        trn_idx = indices[n_val:]
        train_loaders.append(get_client_dataloader(train_dataset, trn_idx, batch_size, shuffle=True))
        val_loaders.append(  get_client_dataloader(train_dataset, val_idx, batch_size, shuffle=False))

    test_loader = get_test_dataloader(test_dataset, batch_size=128)

    pixel_backdoor_loader = None
    semantic_backdoor_loader = None
    effective_semantic_source_label = semantic_source_label
    if not disable_backdoor_eval:
        labels = np.array(test_dataset.targets) if hasattr(test_dataset, "targets") else np.array([test_dataset[i][1] for i in range(len(test_dataset))])
        num_classes = len(np.unique(labels))
        if not (0 <= backdoor_target_label < num_classes):
            raise ValueError(f"backdoor_target_label={backdoor_target_label} is out of range [0, {num_classes - 1}]")
        if not (0 <= semantic_source_label < num_classes):
            raise ValueError(f"semantic_source_label={semantic_source_label} is out of range [0, {num_classes - 1}]")

        semantic_source = semantic_source_label
        if semantic_source == backdoor_target_label:
            semantic_source = (backdoor_target_label + 1) % num_classes
            print(
                f"  [Backdoor] semantic_source_label matched target ({backdoor_target_label}); "
                f"using source={semantic_source} instead."
            )
        effective_semantic_source_label = semantic_source

        pixel_backdoor_loader, semantic_backdoor_loader = get_backdoor_test_loaders(
            test_dataset=test_dataset,
            dataset_name=dataset,
            batch_size=128,
            pixel_target_label=backdoor_target_label,
            pixel_trigger_size=pixel_trigger_size,
            pixel_trigger_value=pixel_trigger_value,
            semantic_source_label=semantic_source,
            semantic_target_label=backdoor_target_label,
        )
        print(
            f"  [Backdoor] Pixel eval samples: {len(pixel_backdoor_loader.dataset)} | "
            f"Semantic eval samples: {len(semantic_backdoor_loader.dataset)}"
        )

    # ── Metrics Tracker ───────────────────────
    os.makedirs(results_dir, exist_ok=True)
    tracker = MetricsTracker(save_dir=results_dir, experiment_name=exp_name)

    # ── Build Flower Apps ─────────────────────
    print("[2/4] Building Flower ClientApp and ServerApp...")

    client_app = build_client_app(
        train_loaders = train_loaders,
        val_loaders   = val_loaders,
        dataset       = dataset,
        local_epochs  = local_epochs,
        learning_rate = learning_rate,
        momentum      = momentum,
        device        = client_device,
    )

    server_app = build_server_app(
        test_loader       = test_loader,
        tracker           = tracker,
        dataset           = dataset,
        num_rounds        = num_rounds,
        num_clients       = num_clients,
        pixel_backdoor_loader = pixel_backdoor_loader,
        semantic_backdoor_loader = semantic_backdoor_loader,
        fraction_fit      = fraction_fit,
        fraction_evaluate = 1.0,
        min_fit_clients   = max(2, int(fraction_fit * num_clients)),
        device            = server_device,
    )

    # ── Run Simulation ────────────────────────
    print(f"[3/4] Starting FL simulation ({num_rounds} rounds)...\n")
    start_time = time.time()

    run_simulation(
        server_app  = server_app,
        client_app  = client_app,
        num_supernodes = num_clients,
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": client_num_gpus}},
    )

    elapsed = time.time() - start_time
    print(f"\n  Simulation completed in {elapsed:.1f}s")

    # ── Save Results ──────────────────────────
    print("[4/4] Saving results...")
    tracker.save_csv()

    # Compute communication cost
    model    = get_model(dataset)
    comm_mb  = compute_comm_cost_mb(
        model,
        num_clients  = int(fraction_fit * num_clients),
        num_rounds   = num_rounds,
    )

    summary = tracker.summary()
    summary.update({
        "method":      "FedAvg",
        "dataset":     dataset.upper(),
        "num_clients": num_clients,
        "num_rounds":  num_rounds,
        "alpha":       alpha_str,
        "comm_mb":     f"{comm_mb:.1f}",
        "client_num_gpus": client_num_gpus,
        "backdoor_target_label": backdoor_target_label,
        "semantic_source_label": effective_semantic_source_label,
        "pixel_trigger_size": pixel_trigger_size,
    })

    return summary


# ─────────────────────────────────────────────
# Run ALL required configurations
# ─────────────────────────────────────────────
def run_all_experiments(
    results_dir: str = "../results",
    client_num_gpus: float = 0.0,
    backdoor_target_label: int = 0,
    semantic_source_label: int = 1,
    pixel_trigger_size: int = 3,
    pixel_trigger_value: float = 1.0,
    disable_backdoor_eval: bool = False,
):
    """
    Run all configurations required by the course:
    - Datasets: MNIST, FashionMNIST, CIFAR-10
    - Clients:  10, 50, 100
    - Alpha:    0.01, 0.1, 0.5, 1.0, iid
    """
    # Course-required configurations
    DATASETS    = ["mnist", "fmnist", "cifar10"]
    NUM_CLIENTS = [10, 50, 100]
    ALPHAS      = [0.01, 0.1, 0.5, 1.0, "iid"]

    # Adjust rounds per dataset (more complex datasets need more rounds)
    ROUNDS = {"mnist": 50, "fmnist": 100, "cifar10": 150}

    all_results = []

    for dataset in DATASETS:
        dataset_trackers = {}

        for num_clients in NUM_CLIENTS:
            iid_noniid_results = {}

            for alpha in ALPHAS:
                result = run_experiment(
                    dataset     = dataset,
                    num_clients = num_clients,
                    num_rounds  = ROUNDS[dataset],
                    alpha       = alpha,
                    client_num_gpus = client_num_gpus,
                    backdoor_target_label = backdoor_target_label,
                    semantic_source_label = semantic_source_label,
                    pixel_trigger_size = pixel_trigger_size,
                    pixel_trigger_value = pixel_trigger_value,
                    disable_backdoor_eval = disable_backdoor_eval,
                    results_dir = results_dir,
                )
                all_results.append(result)
                iid_noniid_results[str(alpha)] = result["final_accuracy"]

            # IID vs Non-IID comparison plot per dataset+clients config
            plot_save_path = os.path.join(
                results_dir,
                f"figure4_iid_noniid_{dataset}_c{num_clients}.png"
            )
            plot_iid_vs_noniid(
                results    = iid_noniid_results,
                save_path  = plot_save_path,
                title      = f"IID vs Non-IID: {dataset.upper()} ({num_clients} clients)",
            )

    # Print summary table
    print_results_table(all_results)

    # Save combined results CSV
    import csv
    combined_path = os.path.join(results_dir, "fedavg_all_results.csv")
    if all_results:
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"\nAll results saved to: {combined_path}")

    return all_results


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="FedAvg Baseline for FL Backdoor Defense Research")

    parser.add_argument("--dataset",     type=str,   default="cifar10",
                        choices=["mnist", "fmnist", "fashionmnist", "cifar10", "cifar100"],
                        help="Dataset to use")
    parser.add_argument("--num_clients", type=int,   default=10,
                        help="Total number of FL clients (try: 10, 50, 100)")
    parser.add_argument("--num_rounds",  type=int,   default=100,
                        help="Number of FL communication rounds")
    parser.add_argument("--alpha",       type=str,   default="0.5",
                        help="Dirichlet alpha for non-IID partitioning. Use 'iid' for IID.")
    parser.add_argument("--local_epochs",type=int,   default=5,
                        help="Local training epochs per round (course requires 5)")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Local batch size (course requires 32)")
    parser.add_argument("--lr",          type=float, default=0.01,
                        help="Learning rate (course requires 0.01)")
    parser.add_argument("--momentum",    type=float, default=0.9,
                        help="SGD momentum (course requires 0.9)")
    parser.add_argument("--fraction_fit",type=float, default=0.5,
                        help="Fraction of clients per round (course requires 0.5)")
    parser.add_argument("--client_num_gpus", type=float, default=0.0,
                        help="GPU resources per Ray client worker (set >0 to enable client-side CUDA)")
    parser.add_argument("--backdoor_target_label", type=int, default=0,
                        help="Target label used to compute backdoor attack success rate (ASR)")
    parser.add_argument("--semantic_source_label", type=int, default=1,
                        help="Source class used for semantic backdoor ASR evaluation")
    parser.add_argument("--pixel_trigger_size", type=int, default=3,
                        help="Square trigger size (in pixels) for pixel-pattern backdoor evaluation")
    parser.add_argument("--pixel_trigger_value", type=float, default=1.0,
                        help="Raw trigger pixel value before normalization (0..1)")
    parser.add_argument("--disable_backdoor_eval", action="store_true",
                        help="Disable per-round pixel/semantic backdoor evaluation")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (course requires 42)")
    parser.add_argument("--results_dir", type=str,   default="../results",
                        help="Directory to save results and plots")
    parser.add_argument("--run_all",     action="store_true",
                        help="Run all required configurations automatically")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_all:
        print("\n Running ALL required experimental configurations...\n")
        run_all_experiments(
            results_dir=args.results_dir,
            client_num_gpus=args.client_num_gpus,
            backdoor_target_label=args.backdoor_target_label,
            semantic_source_label=args.semantic_source_label,
            pixel_trigger_size=args.pixel_trigger_size,
            pixel_trigger_value=args.pixel_trigger_value,
            disable_backdoor_eval=args.disable_backdoor_eval,
        )
    else:
        # Parse alpha
        alpha = args.alpha if args.alpha.lower() == "iid" else float(args.alpha)

        result = run_experiment(
            dataset      = args.dataset,
            num_clients  = args.num_clients,
            num_rounds   = args.num_rounds,
            alpha        = alpha,
            local_epochs = args.local_epochs,
            batch_size   = args.batch_size,
            learning_rate= args.lr,
            momentum     = args.momentum,
            fraction_fit = args.fraction_fit,
            client_num_gpus = args.client_num_gpus,
            backdoor_target_label = args.backdoor_target_label,
            semantic_source_label = args.semantic_source_label,
            pixel_trigger_size = args.pixel_trigger_size,
            pixel_trigger_value = args.pixel_trigger_value,
            disable_backdoor_eval = args.disable_backdoor_eval,
            seed         = args.seed,
            results_dir  = args.results_dir,
        )

        print("\n Final Summary:")
        print_results_table([result])
