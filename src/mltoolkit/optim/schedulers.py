import torch

def make_optimizer(model, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_scheduler(optimizer, epochs: int):
    # cosine schedule
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
