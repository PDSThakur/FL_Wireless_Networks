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
from typing import List

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
    build_poisoned_client_train_loader,
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


def select_malicious_client_ids(num_clients: int, malicious_frac: float, seed: int) -> List[int]:
    """Select malicious client ids deterministically from [0, num_clients)."""
    if malicious_frac <= 0.0:
        return []
    n = int(np.floor(malicious_frac * num_clients))
    if n == 0:
        n = 1
    n = min(n, num_clients)
    rng = np.random.RandomState(seed)
    selected = rng.choice(np.arange(num_clients), size=n, replace=False).tolist()
    return sorted(int(x) for x in selected)


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
    attack_type: str = "none",
    malicious_frac: float = 0.0,
    poison_rate: float = 0.0,
    defense_enabled: bool = False,
    defense_max_samples: int = 64,
    defense_pca_components: int = 5,
    defense_grad_steps: int = 3,
    defense_grad_step_size: float = 0.01,
    defense_mad_threshold: float = 2.5,
    defense_temporal_alpha: float = 0.3,
    defense_expected_malicious_frac: float = 0.1,
    defense_bayes_prior: float = 0.1,
    defense_sigmoid_temperature: float = 1.0,
    defense_min_trust_weight: float = 0.01,
    defense_detector_momentum: float = 0.8,
    defense_bayes_blend: float = 0.6,
    defense_adaptive_threshold: bool = True,
    defense_cv_folds: int = 3,
    defense_hard_filter: bool = False,
    defense_flag_threshold: float = 0.5,
    min_fit_clients: int = 2,
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
    if defense_enabled:
        print(
            "  Defense:    DifFense+Adaptive+Temporal+Ensemble+Bayesian "
            f"| samples={defense_max_samples} pca={defense_pca_components} "
            f"steps={defense_grad_steps} lr={defense_grad_step_size} th={defense_mad_threshold} "
            f"| alpha={defense_temporal_alpha} prior={defense_bayes_prior} "
            f"| adaptive={int(defense_adaptive_threshold)} cv={defense_cv_folds} "
            f"| hard_filter={int(defense_hard_filter)} flag_th={defense_flag_threshold:.2f} "
            f"| min_fit_clients={max(2, min(int(min_fit_clients), num_clients))}"
        )
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

    attack_type = attack_type.lower()
    if attack_type not in ("none", "pixel", "semantic"):
        raise ValueError(f"attack_type must be one of: none, pixel, semantic (got {attack_type})")
    if not (0.0 <= malicious_frac <= 1.0):
        raise ValueError(f"malicious_frac must be in [0,1], got {malicious_frac}")
    if not (0.0 <= poison_rate <= 1.0):
        raise ValueError(f"poison_rate must be in [0,1], got {poison_rate}")
    if not (0.0 <= defense_flag_threshold <= 1.0):
        raise ValueError(f"defense_flag_threshold must be in [0,1], got {defense_flag_threshold}")

    train_labels = np.array(train_dataset.targets) if hasattr(train_dataset, "targets") else np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    train_num_classes = len(np.unique(train_labels))
    semantic_source_for_attack = semantic_source_label
    if attack_type == "semantic" and semantic_source_for_attack == backdoor_target_label:
        semantic_source_for_attack = (backdoor_target_label + 1) % train_num_classes
        print(
            f"  [Attack] semantic_source_label matched target ({backdoor_target_label}); "
            f"using source={semantic_source_for_attack} for training attack."
        )

    malicious_client_ids: List[int] = []
    attack_stats = {}
    if attack_type != "none" and malicious_frac > 0.0 and poison_rate > 0.0:
        malicious_client_ids = select_malicious_client_ids(num_clients, malicious_frac, seed)
        for cid in malicious_client_ids:
            poisoned_loader, stats = build_poisoned_client_train_loader(
                train_loader=train_loaders[cid],
                dataset_name=dataset,
                attack_type=attack_type,
                poison_rate=poison_rate,
                target_label=backdoor_target_label,
                source_label=semantic_source_for_attack,
                trigger_size=pixel_trigger_size,
                trigger_value=pixel_trigger_value,
                seed=seed + cid,
            )
            train_loaders[cid] = poisoned_loader
            attack_stats[cid] = stats

        total_poisoned = sum(s["num_poisoned"] for s in attack_stats.values())
        total_eligible = sum(s["num_eligible"] for s in attack_stats.values())
        print(
            f"  [Attack] Type: {attack_type} | Malicious clients: {len(malicious_client_ids)}/{num_clients} "
            f"({malicious_frac:.2f}) | Poisoned samples: {total_poisoned}/{total_eligible}"
        )
        print(f"  [Attack] Malicious client ids: {malicious_client_ids}")
    elif attack_type != "none":
        print(
            "  [Attack] attack_type is set but no poisoning applied "
            f"(malicious_frac={malicious_frac}, poison_rate={poison_rate})."
        )

    test_loader = get_test_dataloader(test_dataset, batch_size=128)

    pixel_backdoor_loader = None
    semantic_backdoor_loader = None
    effective_semantic_source_label = semantic_source_for_attack
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
        defense_enabled = defense_enabled,
        defense_max_samples = defense_max_samples,
        defense_pca_components = defense_pca_components,
        defense_grad_steps = defense_grad_steps,
        defense_grad_step_size = defense_grad_step_size,
        defense_mad_threshold = defense_mad_threshold,
        defense_temporal_alpha = defense_temporal_alpha,
        defense_expected_malicious_frac = defense_expected_malicious_frac,
        defense_bayes_prior = defense_bayes_prior,
        defense_sigmoid_temperature = defense_sigmoid_temperature,
        defense_min_trust_weight = defense_min_trust_weight,
        defense_detector_momentum = defense_detector_momentum,
        defense_bayes_blend = defense_bayes_blend,
        defense_adaptive_threshold = defense_adaptive_threshold,
        defense_cv_folds = defense_cv_folds,
        defense_hard_filter = defense_hard_filter,
        defense_flag_threshold = defense_flag_threshold,
        fraction_fit      = fraction_fit,
        fraction_evaluate = 1.0,
        min_fit_clients   = max(2, min(int(min_fit_clients), num_clients)),
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

    # FPR/FNR calculation (if defense enabled)
    mean_fpr = mean_fnr = None
    if defense_enabled and hasattr(server_app, 'strategy') and hasattr(server_app.strategy, 'get_defense_flagged_clients_per_round'):
        flagged_per_round = server_app.strategy.get_defense_flagged_clients_per_round()
        all_client_ids = set(str(i) for i in range(num_clients))
        malicious_set = set(str(i) for i in malicious_client_ids)
        benign_set = all_client_ids - malicious_set
        fprs, fnrs = [], []
        for flagged in flagged_per_round:
            flagged_set = set(flagged)
            # FPR: benign clients flagged / total benign
            benign_flagged = len(flagged_set & benign_set)
            fpr = benign_flagged / len(benign_set) if benign_set else 0.0
            # FNR: malicious clients NOT flagged / total malicious
            missed_malicious = len(malicious_set - flagged_set)
            fnr = missed_malicious / len(malicious_set) if malicious_set else 0.0
            fprs.append(fpr)
            fnrs.append(fnr)
        mean_fpr = float(np.mean(fprs)) if fprs else 0.0
        mean_fnr = float(np.mean(fnrs)) if fnrs else 0.0
        print(f"[Defense] Mean FPR: {mean_fpr*100:.2f}% | Mean FNR: {mean_fnr*100:.2f}% over {len(flagged_per_round)} rounds.")

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
        "attack_type": attack_type,
        "malicious_frac": malicious_frac,
        "poison_rate": poison_rate,
        "num_malicious_clients": len(malicious_client_ids),
        "malicious_client_ids": ",".join(str(x) for x in malicious_client_ids),
        "defense_enabled": int(defense_enabled),
        "defense_max_samples": defense_max_samples,
        "defense_pca_components": defense_pca_components,
        "defense_grad_steps": defense_grad_steps,
        "defense_grad_step_size": defense_grad_step_size,
        "defense_mad_threshold": defense_mad_threshold,
        "defense_temporal_alpha": defense_temporal_alpha,
        "defense_expected_malicious_frac": defense_expected_malicious_frac,
        "defense_bayes_prior": defense_bayes_prior,
        "defense_sigmoid_temperature": defense_sigmoid_temperature,
        "defense_min_trust_weight": defense_min_trust_weight,
        "defense_detector_momentum": defense_detector_momentum,
        "defense_bayes_blend": defense_bayes_blend,
        "defense_adaptive_threshold": int(defense_adaptive_threshold),
        "defense_cv_folds": defense_cv_folds,
        "defense_hard_filter": int(defense_hard_filter),
        "defense_flag_threshold": defense_flag_threshold,
        "min_fit_clients": max(2, min(int(min_fit_clients), num_clients)),
        "mean_fpr": mean_fpr,
        "mean_fnr": mean_fnr,
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
    attack_type: str = "none",
    malicious_frac: float = 0.0,
    poison_rate: float = 0.0,
    defense_enabled: bool = False,
    defense_max_samples: int = 64,
    defense_pca_components: int = 5,
    defense_grad_steps: int = 3,
    defense_grad_step_size: float = 0.01,
    defense_mad_threshold: float = 2.5,
    defense_temporal_alpha: float = 0.3,
    defense_expected_malicious_frac: float = 0.1,
    defense_bayes_prior: float = 0.1,
    defense_sigmoid_temperature: float = 1.0,
    defense_min_trust_weight: float = 0.01,
    defense_detector_momentum: float = 0.8,
    defense_bayes_blend: float = 0.6,
    defense_adaptive_threshold: bool = True,
    defense_cv_folds: int = 3,
    defense_hard_filter: bool = False,
    defense_flag_threshold: float = 0.5,
    min_fit_clients: int = 2,
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
                    attack_type = attack_type,
                    malicious_frac = malicious_frac,
                    poison_rate = poison_rate,
                    defense_enabled = defense_enabled,
                    defense_max_samples = defense_max_samples,
                    defense_pca_components = defense_pca_components,
                    defense_grad_steps = defense_grad_steps,
                    defense_grad_step_size = defense_grad_step_size,
                    defense_mad_threshold = defense_mad_threshold,
                    defense_temporal_alpha = defense_temporal_alpha,
                    defense_expected_malicious_frac = defense_expected_malicious_frac,
                    defense_bayes_prior = defense_bayes_prior,
                    defense_sigmoid_temperature = defense_sigmoid_temperature,
                    defense_min_trust_weight = defense_min_trust_weight,
                    defense_detector_momentum = defense_detector_momentum,
                    defense_bayes_blend = defense_bayes_blend,
                    defense_adaptive_threshold = defense_adaptive_threshold,
                    defense_cv_folds = defense_cv_folds,
                    defense_hard_filter = defense_hard_filter,
                    defense_flag_threshold = defense_flag_threshold,
                    min_fit_clients = min_fit_clients,
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
    parser.add_argument("--attack_type", type=str, default="none",
                        choices=["none", "pixel", "semantic"],
                        help="Training-time attacker type: none, pixel, or semantic")
    parser.add_argument("--malicious_frac", type=float, default=0.0,
                        help="Fraction of clients acting as attackers")
    parser.add_argument("--poison_rate", type=float, default=0.0,
                        help="Fraction of eligible local samples poisoned on each malicious client")
    parser.add_argument("--defense_enabled", action="store_true",
                        help="Enable advanced DifFense pipeline (adaptive + temporal + ensemble + Bayesian + soft weighting)")
    parser.add_argument("--defense_max_samples", type=int, default=64,
                        help="Number of server images used for differential testing")
    parser.add_argument("--defense_pca_components", type=int, default=5,
                        help="PCA components for model-behavior embeddings")
    parser.add_argument("--defense_grad_steps", type=int, default=3,
                        help="Gradient ascent steps to generate differential inputs")
    parser.add_argument("--defense_grad_step_size", type=float, default=0.01,
                        help="Step size for differential-input gradient ascent")
    parser.add_argument("--defense_mad_threshold", type=float, default=2.5,
                        help="Threshold on two-step MAD normalized deviation")
    parser.add_argument("--defense_temporal_alpha", type=float, default=0.3,
                        help="EMA decay for temporal suspicion memory (0..1)")
    parser.add_argument("--defense_expected_malicious_frac", type=float, default=0.1,
                        help="Expected malicious fraction used by ensemble detectors")
    parser.add_argument("--defense_bayes_prior", type=float, default=0.1,
                        help="Prior malicious probability for Bayesian posterior")
    parser.add_argument("--defense_sigmoid_temperature", type=float, default=1.0,
                        help="Temperature for soft-weight sigmoid around adaptive threshold")
    parser.add_argument("--defense_min_trust_weight", type=float, default=0.01,
                        help="Lower bound on client trust weight during aggregation")
    parser.add_argument("--defense_detector_momentum", type=float, default=0.8,
                        help="EMA momentum for detector reliability updates")
    parser.add_argument("--defense_bayes_blend", type=float, default=0.6,
                        help="Blend ratio for Bayesian posterior vs ensemble vote (0..1)")
    parser.add_argument("--defense_disable_adaptive_threshold", action="store_true",
                        help="Disable adaptive valley-seeking thresholding and use fixed MAD threshold")
    parser.add_argument("--defense_cv_folds", type=int, default=3,
                        help="K-fold count for adaptive threshold model selection")
    parser.add_argument("--defense_hard_filter", action="store_true",
                        help="Drop clients from aggregation when malicious probability exceeds defense_flag_threshold")
    parser.add_argument("--defense_flag_threshold", type=float, default=0.5,
                        help="Malicious-probability threshold (0..1) used for soft-flagging and optional hard filtering")
    parser.add_argument("--min_fit_clients", type=int, default=2,
                        help="Minimum number of client updates required to aggregate each round")
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
            attack_type=args.attack_type,
            malicious_frac=args.malicious_frac,
            poison_rate=args.poison_rate,
            defense_enabled=args.defense_enabled,
            defense_max_samples=args.defense_max_samples,
            defense_pca_components=args.defense_pca_components,
            defense_grad_steps=args.defense_grad_steps,
            defense_grad_step_size=args.defense_grad_step_size,
            defense_mad_threshold=args.defense_mad_threshold,
            defense_temporal_alpha=args.defense_temporal_alpha,
            defense_expected_malicious_frac=args.defense_expected_malicious_frac,
            defense_bayes_prior=args.defense_bayes_prior,
            defense_sigmoid_temperature=args.defense_sigmoid_temperature,
            defense_min_trust_weight=args.defense_min_trust_weight,
            defense_detector_momentum=args.defense_detector_momentum,
            defense_bayes_blend=args.defense_bayes_blend,
            defense_adaptive_threshold=not args.defense_disable_adaptive_threshold,
            defense_cv_folds=args.defense_cv_folds,
            defense_hard_filter=args.defense_hard_filter,
            defense_flag_threshold=args.defense_flag_threshold,
            min_fit_clients=args.min_fit_clients,
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
            attack_type = args.attack_type,
            malicious_frac = args.malicious_frac,
            poison_rate = args.poison_rate,
            defense_enabled = args.defense_enabled,
            defense_max_samples = args.defense_max_samples,
            defense_pca_components = args.defense_pca_components,
            defense_grad_steps = args.defense_grad_steps,
            defense_grad_step_size = args.defense_grad_step_size,
            defense_mad_threshold = args.defense_mad_threshold,
            defense_temporal_alpha = args.defense_temporal_alpha,
            defense_expected_malicious_frac = args.defense_expected_malicious_frac,
            defense_bayes_prior = args.defense_bayes_prior,
            defense_sigmoid_temperature = args.defense_sigmoid_temperature,
            defense_min_trust_weight = args.defense_min_trust_weight,
            defense_detector_momentum = args.defense_detector_momentum,
            defense_bayes_blend = args.defense_bayes_blend,
            defense_adaptive_threshold = not args.defense_disable_adaptive_threshold,
            defense_cv_folds = args.defense_cv_folds,
            defense_hard_filter = args.defense_hard_filter,
            defense_flag_threshold = args.defense_flag_threshold,
            min_fit_clients = args.min_fit_clients,
            seed         = args.seed,
            results_dir  = args.results_dir,
        )

        print("\n Final Summary:")
        print_results_table([result])
