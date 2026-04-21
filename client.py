"""
client.py — Flower ClientApp for FedAvg baseline
"""

import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from model import get_model
from data  import load_full_dataset, get_partition, get_client_dataloader
from utils import train, evaluate


# ─────────────────────────────────────────────
# Flower NumPy Client
# ─────────────────────────────────────────────
class FedAvgClient(NumPyClient):
    """
    Standard FedAvg client.

    Each client:
    1. Receives global model weights from server
    2. Trains locally for `local_epochs` epochs
    3. Returns updated weights + metrics
    """

    def __init__(
        self,
        client_id: int,
        train_loader,
        val_loader,
        dataset: str,
        local_epochs: int,
        learning_rate: float,
        momentum: float,
        device: torch.device,
    ):
        self.client_id    = client_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.dataset      = dataset
        self.local_epochs = local_epochs
        self.lr           = learning_rate
        self.momentum     = momentum
        self.device       = device
        self.model        = get_model(dataset).to(device)

    # ── Flower interface ──────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return local model parameters as list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load parameters received from server into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Receive global model → train locally → return updated model.

        Returns:
            (updated_parameters, num_samples, metrics_dict)
        """
        # Load global model weights
        self.set_parameters(parameters)

        # Read config (can be overridden per-round from server)
        epochs = config.get("local_epochs", self.local_epochs)
        lr     = config.get("learning_rate", self.lr)

        # Local training
        loss, acc = train(
            model       = self.model,
            dataloader  = self.train_loader,
            epochs      = epochs,
            learning_rate = lr,
            momentum    = self.momentum,
            device      = self.device,
        )

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"train_loss": loss, "train_accuracy": acc},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local validation data.

        Returns:
            (loss, num_samples, metrics_dict)
        """
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.device)
        return (
            float(loss),
            len(self.val_loader.dataset),
            {"accuracy": float(acc)},
        )


# ─────────────────────────────────────────────
# ClientApp factory (used by Flower simulation)
# ─────────────────────────────────────────────
def make_client_fn(
    train_loaders,
    val_loaders,
    dataset: str,
    local_epochs: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
):
    """
    Returns a client_fn closure that Flower calls to create client instances.
    Each client is identified by a partition_id (= client index).
    """

    def client_fn(context: Context) -> NumPyClient:
        # Flower passes partition_id via context
        partition_id = int(context.node_config["partition-id"])

        return FedAvgClient(
            client_id    = partition_id,
            train_loader = train_loaders[partition_id],
            val_loader   = val_loaders[partition_id],
            dataset      = dataset,
            local_epochs = local_epochs,
            learning_rate= learning_rate,
            momentum     = momentum,
            device       = device,
        )

    return client_fn


def build_client_app(
    train_loaders,
    val_loaders,
    dataset: str,
    local_epochs: int   = 5,
    learning_rate: float = 0.01,
    momentum: float     = 0.9,
    device: torch.device = None,
) -> ClientApp:
    """Build and return a Flower ClientApp."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_fn = make_client_fn(
        train_loaders, val_loaders, dataset,
        local_epochs, learning_rate, momentum, device
    )
    return ClientApp(client_fn=client_fn)
