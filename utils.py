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
        # Keep metric columns aligned even when some keys appear only in later rounds.
        row_idx = len(self.history["round"])
        self.history["round"].append(round_num)

        # Add a default value for this round to all previously seen metric columns.
        for key in list(self.history.keys()):
            if key == "round":
                continue
            # Heal any historical mismatch before appending.
            if len(self.history[key]) < row_idx:
                self.history[key].extend([np.nan] * (row_idx - len(self.history[key])))
            self.history[key].append(np.nan)

        # Fill this round's values.
        for key, value in metrics.items():
            if key not in self.history:
                # Metric introduced mid-run: backfill previous rounds.
                self.history[key] = [np.nan] * row_idx + [value]
            else:
                self.history[key][row_idx] = value

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
            # Use number of logged rounds as canonical row count.
            num_rows = len(self.history.get("round", []))
            for i in range(num_rows):
                row = {k: (self.history[k][i] if i < len(self.history[k]) else "") for k in keys}
                writer.writerow(row)
        print(f"  Saved metrics → {path}")

    def summary(self) -> dict:
        """Return a summary dict of final metrics."""
        accs = self.history.get("global_accuracy", [0])
        losses = self.history.get("global_loss", [0])
        out = {
            "final_accuracy": accs[-1] if accs else 0,
            "final_loss":     losses[-1] if losses else 0,
            "convergence_round": self.get_convergence_round(),
            "total_rounds":   len(self.history.get("round", [])),
        }
        pixel_accs = self.history.get("pixel_backdoor_accuracy", [])
        semantic_accs = self.history.get("semantic_backdoor_accuracy", [])
        if pixel_accs:
            out["final_pixel_backdoor_accuracy"] = pixel_accs[-1]
        if semantic_accs:
            out["final_semantic_backdoor_accuracy"] = semantic_accs[-1]
        defense_flags = self.history.get("defense_flagged_clients", [])
        if defense_flags:
            out["final_defense_flagged_clients"] = defense_flags[-1]
            out["avg_defense_flagged_clients"] = float(np.mean(defense_flags))
        detection_metrics = [
            "defense_precision",
            "defense_recall",
            "defense_f1",
            "defense_false_positive_rate",
            "defense_true_positive_rate",
            "defense_true_positives",
            "defense_false_positives",
            "defense_false_negatives",
            "defense_true_negatives",
            "defense_malicious_participants",
            "defense_benign_participants",
        ]
        for key in detection_metrics:
            values = self.history.get(key, [])
            if not values:
                continue
            numeric = np.array(values, dtype=float)
            valid = numeric[~np.isnan(numeric)]
            if valid.size == 0:
                continue
            out[f"final_{key}"] = float(valid[-1])
            out[f"avg_{key}"] = float(np.mean(valid))
        return out


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


def save_rows_csv(rows: List[Dict], path: str) -> None:
    """Save a list of dict rows to CSV using union of all keys."""
    if not rows:
        return
    key_order = list(rows[0].keys())
    key_set = set(key_order)
    for row in rows[1:]:
        for key in row.keys():
            if key not in key_set:
                key_order.append(key)
                key_set.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=key_order)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in key_order})
    print(f"  Saved CSV -> {path}")


def _to_float(value) -> float:
    """Best-effort float conversion returning NaN for invalid values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_results_mean_std(
    rows: List[Dict],
    group_keys: List[str],
    metric_keys: List[str],
) -> List[Dict]:
    """
    Aggregate rows by group keys and compute mean/std for selected metrics.
    Returns one row per unique group.
    """
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in group_keys)].append(row)

    out_rows: List[Dict] = []
    for group_values, group_rows in grouped.items():
        out = {k: v for k, v in zip(group_keys, group_values)}
        out["num_runs"] = len(group_rows)
        for key in metric_keys:
            vals = np.array([_to_float(r.get(key, np.nan)) for r in group_rows], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                out[f"{key}_mean"] = ""
                out[f"{key}_std"] = ""
            else:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
        out_rows.append(out)
    return out_rows


def plot_sweep_metric(
    rows: List[Dict],
    x_key: str,
    series_key: str,
    metric_key: str,
    save_path: str,
    title: str,
    y_label: str,
):
    """
    Plot aggregated sweep metric as line plot:
    x-axis = x_key, one line per series_key.
    """
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    series_values = sorted({str(r.get(series_key, "")) for r in rows}, key=lambda x: _to_float(x))
    for i, s_val in enumerate(series_values):
        subset = [r for r in rows if str(r.get(series_key, "")) == s_val]
        subset_sorted = sorted(subset, key=lambda r: _to_float(r.get(x_key, "")))
        xs = [_to_float(r.get(x_key, "")) for r in subset_sorted]
        ys = [_to_float(r.get(f"{metric_key}_mean", np.nan)) for r in subset_sorted]
        yerr = [_to_float(r.get(f"{metric_key}_std", np.nan)) for r in subset_sorted]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=2,
            capsize=4,
            label=f"{series_key}={s_val}",
        )

    _style_axis(ax, x_key, y_label, title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Figure -> {save_path}")


def write_sweep_summary_text(
    raw_rows: List[Dict],
    aggregated_rows: List[Dict],
    path: str,
) -> None:
    """Write a short, paper-friendly summary text for sweep experiments."""
    if not raw_rows:
        return

    raw_acc = np.array([_to_float(r.get("final_accuracy", np.nan)) for r in raw_rows], dtype=float)
    raw_acc = raw_acc[~np.isnan(raw_acc)]

    raw_pixel = np.array(
        [_to_float(r.get("final_pixel_backdoor_accuracy", np.nan)) for r in raw_rows], dtype=float
    )
    raw_pixel = raw_pixel[~np.isnan(raw_pixel)]

    raw_semantic = np.array(
        [_to_float(r.get("final_semantic_backdoor_accuracy", np.nan)) for r in raw_rows], dtype=float
    )
    raw_semantic = raw_semantic[~np.isnan(raw_semantic)]

    raw_f1 = np.array([_to_float(r.get("final_defense_f1", np.nan)) for r in raw_rows], dtype=float)
    raw_f1 = raw_f1[~np.isnan(raw_f1)]

    best_acc_row = max(raw_rows, key=lambda r: _to_float(r.get("final_accuracy", -np.inf)))
    best_f1_row = max(raw_rows, key=lambda r: _to_float(r.get("final_defense_f1", -np.inf)))

    lines = [
        "Sweep Summary",
        "=============",
        f"Total runs: {len(raw_rows)}",
        "",
        (
            f"Final clean accuracy (mean +/- std): "
            f"{float(np.mean(raw_acc)):.4f} +/- {float(np.std(raw_acc)):.4f}"
            if raw_acc.size > 0
            else "Final clean accuracy: N/A"
        ),
        (
            f"Final pixel ASR (mean +/- std): "
            f"{float(np.mean(raw_pixel)):.4f} +/- {float(np.std(raw_pixel)):.4f}"
            if raw_pixel.size > 0
            else "Final pixel ASR: N/A"
        ),
        (
            f"Final semantic ASR (mean +/- std): "
            f"{float(np.mean(raw_semantic)):.4f} +/- {float(np.std(raw_semantic)):.4f}"
            if raw_semantic.size > 0
            else "Final semantic ASR: N/A"
        ),
        (
            f"Final defense F1 (mean +/- std): "
            f"{float(np.mean(raw_f1)):.4f} +/- {float(np.std(raw_f1)):.4f}"
            if raw_f1.size > 0
            else "Final defense F1: N/A"
        ),
        "",
        "Best clean-accuracy run:",
        (
            f"  seed={best_acc_row.get('seed', '')}, "
            f"malicious_frac={best_acc_row.get('malicious_frac', '')}, "
            f"poison_rate={best_acc_row.get('poison_rate', '')}, "
            f"final_accuracy={_to_float(best_acc_row.get('final_accuracy', np.nan)):.4f}, "
            f"final_semantic_ASR={_to_float(best_acc_row.get('final_semantic_backdoor_accuracy', np.nan)):.4f}, "
            f"final_pixel_ASR={_to_float(best_acc_row.get('final_pixel_backdoor_accuracy', np.nan)):.4f}"
        ),
        "",
        "Best defense-F1 run:",
        (
            f"  seed={best_f1_row.get('seed', '')}, "
            f"malicious_frac={best_f1_row.get('malicious_frac', '')}, "
            f"poison_rate={best_f1_row.get('poison_rate', '')}, "
            f"final_defense_precision={_to_float(best_f1_row.get('final_defense_precision', np.nan)):.4f}, "
            f"final_defense_recall={_to_float(best_f1_row.get('final_defense_recall', np.nan)):.4f}, "
            f"final_defense_f1={_to_float(best_f1_row.get('final_defense_f1', np.nan)):.4f}"
        ),
        "",
        f"Aggregated config rows: {len(aggregated_rows)}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved summary text -> {path}")
