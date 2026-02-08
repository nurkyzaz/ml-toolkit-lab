from dataclasses import dataclass
from typing import Callable, Optional, Dict
import torch
from tqdm import tqdm

from mltoolkit.utils.metrics import accuracy, binary_accuracy

@dataclass
class TrainState:
    epoch: int
    step: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: str,
    grad_clip: float,
    task: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)

        if task == "cv":
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy(logits, y)
            bs = x.size(0)
        elif task == "nlp":
            x, lengths, y = batch
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            acc = accuracy(logits, y)
            bs = x.size(0)
        else:
            raise ValueError(f"Unknown task: {task}")

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device: str, task: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        if task == "cv":
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            acc = accuracy(logits, y)
            bs = x.size(0)
        elif task == "nlp":
            x, lengths, y = batch
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            acc = accuracy(logits, y)
            bs = x.size(0)
        else:
            raise ValueError(f"Unknown task: {task}")

        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}
