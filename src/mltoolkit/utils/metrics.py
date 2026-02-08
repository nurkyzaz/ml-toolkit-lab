import torch

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) > 0.5).long()
    return (preds == y).float().mean().item()
