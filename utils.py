"""
utils.py — Training loop, evaluation, metrics tracking, and plotting
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────
def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train model locally for given number of epochs.

    Returns:
        (avg_loss, accuracy) over the training data
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0

    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total   if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Returns:
        (loss, accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total   if total > 0 else 0.0
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Metrics Tracker
# ─────────────────────────────────────────────
class MetricsTracker:
    """
    Tracks and persists all required metrics per round.
    Matches the mandatory reporting format from course requirements.
    """

    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir  = save_dir
        self.name      = experiment_name
        self.history: Dict[str, List] = defaultdict(list)
        os.makedirs(save_dir, exist_ok=True)

    def record(self, round_num: int, **metrics):
        """Record metrics for a given round."""
        self.history["round"].append(round_num)
        for key, value in metrics.items():
            self.history[key].append(value)

    def get_convergence_round(self, threshold: float = 0.80) -> Optional[int]:
        """
        Return first round where global accuracy exceeds threshold.
        Returns None if threshold never reached.
        """
        accs   = self.history.get("global_accuracy", [])
        rounds = self.history.get("round", [])
        for r, a in zip(rounds, accs):
            if a >= threshold:
                return r
        return None

    def save_csv(self):
        """Save all metrics to a CSV file."""
        path = os.path.join(self.save_dir, f"{self.name}.csv")
        if not self.history:
            return
        keys = list(self.history.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            num_rows = len(self.history[keys[0]])
            for i in range(num_rows):
                row = {k: self.history[k][i] for k in keys}
                writer.writerow(row)
        print(f"  Saved metrics → {path}")

    def summary(self) -> dict:
        """Return a summary dict of final metrics."""
        accs = self.history.get("global_accuracy", [0])
        losses = self.history.get("global_loss", [0])
        return {
            "final_accuracy": accs[-1] if accs else 0,
            "final_loss":     losses[-1] if losses else 0,
            "convergence_round": self.get_convergence_round(),
            "total_rounds":   len(self.history.get("round", [])),
        }


# ─────────────────────────────────────────────
# Communication Cost Calculator
# ─────────────────────────────────────────────
def compute_comm_cost_mb(model: nn.Module, num_clients: int, num_rounds: int) -> float:
    """
    Estimate total communication cost in MB.
    Formula: model_size_mb × clients_per_round × rounds × 2 (upload + download)
    """
    total_params = sum(p.numel() for p in model.parameters())
    # float32 = 4 bytes per param
    model_size_mb = (total_params * 4) / (1024 ** 2)
    # ×2 because both upload (client→server) and download (server→client)
    total_mb = model_size_mb * num_clients * num_rounds * 2
    return total_mb


# ─────────────────────────────────────────────
# Plotting Functions (Mandatory plots)
# ─────────────────────────────────────────────
COLORS  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "s", "^", "D", "v"]


def _style_axis(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title,  fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)


def plot_accuracy_vs_rounds(
    trackers: Dict[str, "MetricsTracker"],
    save_path: str,
    title: str = "Global Accuracy vs Communication Rounds",
):
    """Figure 1 — Mandatory plot: accuracy vs rounds for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, tracker) in enumerate(trackers.items()):
        rounds = tracker.history["round"]
        accs   = tracker.history["global_accuracy"]
        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(rounds, [a * 100 for a in accs],
                label=label, color=color, marker=marker,
                markevery=max(1, len(rounds)//10), linewidth=2, markersize=6)
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.6, label="80% threshold")
    _style_axis(ax, "Communication Round", "Global Test Accuracy (%)", title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure → {save_path}")


def plot_loss_vs_rounds(
    trackers: Dict[str, "MetricsTracker"],
    save_path: str,
    title: str = "Global Loss vs Communication Rounds",
):
    """Figure 2 — Mandatory plot: loss vs rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, tracker) in enumerate(trackers.items()):
        rounds = tracker.history["round"]
        losses = tracker.history["global_loss"]
        ax.plot(rounds, losses,
                label=label, color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                markevery=max(1, len(rounds)//10), linewidth=2, markersize=6)
    _style_axis(ax, "Communication Round", "Global Test Loss", title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure → {save_path}")


def plot_iid_vs_noniid(
    results: Dict[str, float],
    save_path: str,
    title: str = "IID vs Non-IID: Final Accuracy",
):
    """Figure 4 — IID vs Non-IID comparison bar chart."""
    labels = list(results.keys())
    values = [v * 100 for v in results.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=COLORS[:len(labels)], width=0.5, edgecolor="black")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 105)
    ax.set_xlabel("Data Partitioning (α)", fontsize=12)
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure → {save_path}")


def plot_method_comparison(
    results: Dict[str, Dict],
    save_path: str,
    title: str = "Method Comparison: FedAvg vs DifFense",
):
    """Figure 5 — Side-by-side bar chart comparing methods."""
    methods  = list(results.keys())
    datasets = list(results[methods[0]].keys())
    x        = np.arange(len(datasets))
    width    = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        vals = [results[method].get(ds, 0) * 100 for ds in datasets]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method,
                      color=COLORS[i % len(COLORS)], edgecolor="black")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure → {save_path}")


# ─────────────────────────────────────────────
# Results Table Printer
# ─────────────────────────────────────────────
def print_results_table(rows: List[Dict]):
    """Print a formatted results table to console."""
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}

    header = " | ".join(k.ljust(widths[k]) for k in keys)
    sep    = "-+-".join("-" * widths[k] for k in keys)
    print(f"\n{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        line = " | ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys)
        print(line)
    print(f"{'='*len(header)}\n")
