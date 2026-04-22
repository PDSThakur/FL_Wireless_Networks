"""
data.py — Dataset loading + Dirichlet Non-IID partitioning
Supports: MNIST, FashionMNIST, CIFAR-10, CIFAR-100
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
TRANSFORMS = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "fashionmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]),
    "fmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]),
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]),
    "cifar10_test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
    "cifar100_test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
}

NORM_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.2860,), (0.3530,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


# ─────────────────────────────────────────────
# Load full dataset
# ─────────────────────────────────────────────
def load_full_dataset(dataset: str, data_dir: str = "./data"):
    """Load train + test split for a given dataset name."""
    dataset = dataset.lower()

    if dataset == "mnist":
        train = datasets.MNIST(data_dir, train=True,  download=True, transform=TRANSFORMS["mnist"])
        test  = datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORMS["mnist"])
    elif dataset in ("fashionmnist", "fmnist"):
        train = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=TRANSFORMS["fashionmnist"])
        test  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=TRANSFORMS["fashionmnist"])
    elif dataset == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=TRANSFORMS["cifar10"])
        test  = datasets.CIFAR10(data_dir, train=False, download=True, transform=TRANSFORMS["cifar10_test"])
    elif dataset == "cifar100":
        train = datasets.CIFAR100(data_dir, train=True,  download=True, transform=TRANSFORMS["cifar100"])
        test  = datasets.CIFAR100(data_dir, train=False, download=True, transform=TRANSFORMS["cifar100_test"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train, test


def _dataset_labels(dataset) -> np.ndarray:
    """Return labels as a numpy array for datasets exposing `.targets`."""
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def _normalized_trigger_values(dataset_name: str, trigger_value: float) -> torch.Tensor:
    """
    Convert a raw pixel value (0..1) to the normalized space used by each dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in NORM_STATS:
        raise ValueError(f"Unsupported dataset for trigger normalization: {dataset_name}")
    means, stds = NORM_STATS[dataset_name]
    values = [(trigger_value - m) / s for m, s in zip(means, stds)]
    return torch.tensor(values, dtype=torch.float32)


def _apply_pixel_trigger(
    image: torch.Tensor,
    trigger_values: torch.Tensor,
    trigger_size: int,
) -> torch.Tensor:
    """Apply a square pixel trigger in the bottom-right corner."""
    poisoned = image.clone()
    if poisoned.dim() != 3:
        raise ValueError(f"Expected image shape [C,H,W], got {tuple(poisoned.shape)}")

    channels, height, width = poisoned.shape
    size = max(1, min(trigger_size, height, width))
    h_start = height - size
    w_start = width - size

    if trigger_values.numel() == 1 and channels > 1:
        trigger_values = trigger_values.repeat(channels)
    if trigger_values.numel() != channels:
        raise ValueError(
            f"Trigger channel mismatch: image has {channels} channels but trigger has {trigger_values.numel()} values"
        )

    trigger_values = trigger_values.to(poisoned.dtype).to(poisoned.device)
    for c in range(channels):
        poisoned[c, h_start:, w_start:] = trigger_values[c]
    return poisoned


class PixelPatternBackdoorDataset(Dataset):
    """
    Test-time dataset for pixel-pattern backdoor evaluation.

    Each sample receives a trigger patch and label is forced to `target_label`.
    """

    def __init__(
        self,
        base_dataset,
        dataset_name: str,
        target_label: int,
        trigger_size: int = 3,
        trigger_value: float = 1.0,
        include_target_class: bool = False,
    ):
        self.base_dataset = base_dataset
        self.dataset_name = dataset_name.lower()
        self.target_label = int(target_label)
        self.trigger_size = int(trigger_size)
        self.trigger_values = _normalized_trigger_values(self.dataset_name, trigger_value)
        self.include_target_class = include_target_class

        labels = _dataset_labels(base_dataset)
        if self.include_target_class:
            self.indices = list(range(len(base_dataset)))
        else:
            self.indices = np.where(labels != self.target_label)[0].tolist()
            if len(self.indices) == 0:
                raise ValueError(
                    f"No non-target samples available for pixel ASR with target_label={self.target_label}"
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        poisoned = _apply_pixel_trigger(image, self.trigger_values, self.trigger_size)
        return poisoned, self.target_label


class SemanticBackdoorDataset(Dataset):
    """
    Test-time dataset for semantic backdoor evaluation.

    Uses only source-class samples and changes labels to `target_label`.
    """

    def __init__(
        self,
        base_dataset,
        source_label: int,
        target_label: int,
    ):
        self.base_dataset = base_dataset
        self.source_label = int(source_label)
        self.target_label = int(target_label)

        labels = _dataset_labels(base_dataset)
        self.indices = np.where(labels == self.source_label)[0].tolist()
        if len(self.indices) == 0:
            raise ValueError(
                f"No samples found for semantic source_label={self.source_label}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[self.indices[idx]]
        return image, self.target_label


def get_backdoor_test_loaders(
    test_dataset,
    dataset_name: str,
    batch_size: int = 128,
    pixel_target_label: int = 0,
    pixel_trigger_size: int = 3,
    pixel_trigger_value: float = 1.0,
    semantic_source_label: int = 1,
    semantic_target_label: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build backdoor test loaders for:
    1) pixel-pattern backdoor attack
    2) semantic source-class-to-target attack
    """
    pixel_dataset = PixelPatternBackdoorDataset(
        base_dataset=test_dataset,
        dataset_name=dataset_name,
        target_label=pixel_target_label,
        trigger_size=pixel_trigger_size,
        trigger_value=pixel_trigger_value,
        include_target_class=False,
    )

    semantic_dataset = SemanticBackdoorDataset(
        base_dataset=test_dataset,
        source_label=semantic_source_label,
        target_label=semantic_target_label,
    )

    pixel_loader = DataLoader(pixel_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    semantic_loader = DataLoader(semantic_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return pixel_loader, semantic_loader


# ─────────────────────────────────────────────
# Dirichlet Non-IID Partitioning
# ─────────────────────────────────────────────
def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset indices among clients using Dirichlet distribution.

    Args:
        dataset   : PyTorch dataset with targets attribute
        num_clients: Number of FL clients
        alpha     : Dirichlet concentration parameter
                    - Small alpha (0.01) → very non-IID
                    - Large alpha (100)  → nearly IID
        seed      : Random seed for reproducibility

    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples across clients via Dirichlet
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to actual counts
        proportions = np.array([
            p * len(cls_indices) for p in proportions
        ])
        proportions = np.round(proportions).astype(int)

        # Fix rounding errors — make sure total matches
        diff = len(cls_indices) - proportions.sum()
        proportions[np.argmax(proportions)] += diff

        # Assign indices to clients
        start = 0
        for client_id, count in enumerate(proportions):
            end = start + count
            client_indices[client_id].extend(cls_indices[start:end].tolist())
            start = end

    # Shuffle each client's data
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


def iid_partition(dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Partition dataset indices equally and randomly (IID)."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    splits  = np.array_split(indices, num_clients)
    return [s.tolist() for s in splits]


def get_partition(
    dataset,
    num_clients: int,
    alpha,           # float or "iid"
    seed: int = 42,
) -> List[List[int]]:
    """
    Unified partition function.

    alpha can be:
        - a float (e.g. 0.01, 0.1, 0.5, 1.0) → Dirichlet Non-IID
        - "iid"                                 → uniform IID split
    """
    if str(alpha).lower() == "iid":
        return iid_partition(dataset, num_clients, seed)
    else:
        return dirichlet_partition(dataset, num_clients, float(alpha), seed)


# ─────────────────────────────────────────────
# DataLoader builders
# ─────────────────────────────────────────────
def get_client_dataloader(
    dataset,
    indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Return a DataLoader for a specific client's data subset."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def get_test_dataloader(test_dataset, batch_size: int = 128) -> DataLoader:
    """Return a DataLoader for the global test set."""
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# ─────────────────────────────────────────────
# Quick stats helper (useful for debugging)
# ─────────────────────────────────────────────
def partition_stats(dataset, client_indices: List[List[int]]) -> None:
    """Print basic stats about the partition."""
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    print(f"\n{'─'*50}")
    print(f"Partition Stats: {len(client_indices)} clients, {num_classes} classes")
    print(f"{'─'*50}")
    sizes = [len(idx) for idx in client_indices]
    print(f"  Samples per client — min: {min(sizes)}, max: {max(sizes)}, mean: {np.mean(sizes):.1f}")
    print(f"{'─'*50}\n")
