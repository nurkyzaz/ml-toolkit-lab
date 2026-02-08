from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 2
    log_every: int = 50

    # early stopping
    patience: int = 5
    min_delta: float = 0.0

@dataclass
class DPConfig:
    enabled: bool = False
    epsilon: float = 8.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
