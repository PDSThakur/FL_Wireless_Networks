"""
server.py — Flower ServerApp with FedAvg strategy
Handles global model evaluation and metrics aggregation.
"""

from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import (
    Metrics,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Context,
)

from model import get_model
from utils import evaluate, MetricsTracker


# ─────────────────────────────────────────────
# Metrics aggregation helpers
# ─────────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate client metrics using weighted average by number of samples.
    Used for both fit and evaluate aggregation.
    """
    if not metrics:
        return {}

    total_samples = sum(n for n, _ in metrics)
    aggregated    = {}

    # Get all metric keys from first result
    all_keys = metrics[0][1].keys()
    for key in all_keys:
        weighted_sum = sum(n * m[key] for n, m in metrics if key in m)
        aggregated[key] = weighted_sum / total_samples

    return aggregated


# ─────────────────────────────────────────────
# Custom FedAvg Strategy with server-side eval
# ─────────────────────────────────────────────
class FedAvgWithEval(FedAvg):
    """
    Extends FedAvg to:
    1. Evaluate global model on centralized test set each round
    2. Track and store all metrics via MetricsTracker
    3. Log progress to console
    """

    def __init__(
        self,
        test_loader,
        tracker: MetricsTracker,
        dataset: str,
        device: torch.device,
        pixel_backdoor_loader=None,
        semantic_backdoor_loader=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_loader             = test_loader
        self.pixel_backdoor_loader   = pixel_backdoor_loader
        self.semantic_backdoor_loader = semantic_backdoor_loader
        self.tracker                 = tracker
        self.dataset                 = dataset
        self.device                  = device
        self.model                   = get_model(dataset).to(device)

    def aggregate_fit(
        self,
        server_round: int,
        results: List,
        failures: List,
    ):
        """Aggregate model updates from clients (standard FedAvg)."""
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        return aggregated_params, aggregated_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model on centralized test set.
        Called by Flower after each aggregation round.
        """
        # Load aggregated parameters into model
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), ndarrays)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Evaluate on test set
        loss, acc = evaluate(self.model, self.test_loader, self.device)

        metrics = {
            "global_accuracy": acc,
            "global_loss": loss,
        }

        if self.pixel_backdoor_loader is not None:
            pixel_loss, pixel_asr = evaluate(self.model, self.pixel_backdoor_loader, self.device)
            metrics["pixel_backdoor_loss"] = pixel_loss
            metrics["pixel_backdoor_accuracy"] = pixel_asr

        if self.semantic_backdoor_loader is not None:
            semantic_loss, semantic_asr = evaluate(self.model, self.semantic_backdoor_loader, self.device)
            metrics["semantic_backdoor_loss"] = semantic_loss
            metrics["semantic_backdoor_accuracy"] = semantic_asr

        # Record metrics
        self.tracker.record(round_num=server_round, **metrics)

        # Console logging
        log = (
            f"  [Round {server_round:3d}] "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {acc*100:.2f}%"
        )
        if "pixel_backdoor_accuracy" in metrics:
            log += f" | Pixel ASR: {metrics['pixel_backdoor_accuracy']*100:.2f}%"
        if "semantic_backdoor_accuracy" in metrics:
            log += f" | Semantic ASR: {metrics['semantic_backdoor_accuracy']*100:.2f}%"
        print(log)

        return_metrics = {"accuracy": float(acc)}
        if "pixel_backdoor_accuracy" in metrics:
            return_metrics["pixel_backdoor_accuracy"] = float(metrics["pixel_backdoor_accuracy"])
        if "semantic_backdoor_accuracy" in metrics:
            return_metrics["semantic_backdoor_accuracy"] = float(metrics["semantic_backdoor_accuracy"])
        return float(loss), return_metrics


# ─────────────────────────────────────────────
# Build ServerApp
# ─────────────────────────────────────────────
def build_server_app(
    test_loader,
    tracker: MetricsTracker,
    dataset: str,
    num_rounds: int,
    num_clients: int,
    pixel_backdoor_loader=None,
    semantic_backdoor_loader=None,
    fraction_fit: float     = 0.5,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int    = 2,
    device: torch.device    = None,
) -> ServerApp:
    """
    Build and return a Flower ServerApp with FedAvg strategy.

    Args:
        test_loader      : Global centralized test DataLoader
        tracker          : MetricsTracker instance to record results
        dataset          : Dataset name (for model initialization)
        num_rounds       : Total FL communication rounds
        num_clients      : Total number of clients
        pixel_backdoor_loader: DataLoader for pixel trigger backdoor evaluation
        semantic_backdoor_loader: DataLoader for semantic backdoor evaluation
        fraction_fit     : Fraction of clients selected per round (0.5 = 50%)
        fraction_evaluate: Fraction evaluated per round
        min_fit_clients  : Minimum clients required for a round to proceed
        device           : torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize global model and convert to Flower Parameters
    global_model    = get_model(dataset).to(device)
    initial_params  = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    )

    # Build strategy
    strategy = FedAvgWithEval(
        test_loader             = test_loader,
        pixel_backdoor_loader   = pixel_backdoor_loader,
        semantic_backdoor_loader = semantic_backdoor_loader,
        tracker                 = tracker,
        dataset                 = dataset,
        device                  = device,
        # Standard FedAvg parameters
        fraction_fit            = fraction_fit,
        fraction_evaluate       = fraction_evaluate,
        min_fit_clients         = min_fit_clients,
        min_evaluate_clients    = min_fit_clients,
        min_available_clients   = min_fit_clients,
        initial_parameters      = initial_params,
        # Aggregate client metrics
        fit_metrics_aggregation_fn      = weighted_average,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
