from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_cifar10_loaders(
    root: str = "data/cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root=str(root_path), train=True, download=True, transform=train_tfms)
    test_ds = datasets.CIFAR10(root=str(root_path), train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
