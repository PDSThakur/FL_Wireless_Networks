"""
model.py — Model definitions for all datasets
Supports: MNIST, FashionMNIST, CIFAR-10, CIFAR-100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# MNIST / FashionMNIST  →  Simple 3-layer CNN
# (matches the paper's CNN for F-MNIST)
# ─────────────────────────────────────────────
class CNNMnist(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)


# ─────────────────────────────────────────────
# CIFAR-10  →  Lightweight ResNet-18
# (matches the paper exactly)
# ─────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """Lightweight ResNet-18 for CIFAR-10/100 (~2.7M params as in the paper)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ─────────────────────────────────────────────
# CIFAR-100  →  Same ResNet-18, more output classes
# ─────────────────────────────────────────────
class ResNet18Cifar100(ResNet18):
    def __init__(self):
        super().__init__(num_classes=100)


# ─────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────
def get_model(dataset: str) -> nn.Module:
    """Return the correct model for a given dataset name."""
    dataset = dataset.lower()
    if dataset in ("mnist", "fashionmnist", "fmnist"):
        num_classes = 10
        return CNNMnist(num_classes)
    elif dataset == "cifar10":
        return ResNet18(num_classes=10)
    elif dataset == "cifar100":
        return ResNet18(num_classes=100)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. "
                         f"Choose from: mnist, fashionmnist, cifar10, cifar100")
