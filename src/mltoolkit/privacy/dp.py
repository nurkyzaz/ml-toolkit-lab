from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DPEngine:
    privacy_engine: object
    epsilon: float
    delta: float

def maybe_make_private(
    model,
    optimizer,
    data_loader,
    *,
    enabled: bool,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
) -> Tuple[object, object, object, Optional[DPEngine]]:
    """
    If enabled, wraps (model, optimizer, data_loader) with Opacus DP-SGD.
    If opacus isn't installed, raises a helpful error.
    """
    if not enabled:
        return model, optimizer, data_loader, None

    try:
        from opacus import PrivacyEngine
    except Exception as e:
        raise RuntimeError(
            "Differential Privacy requested but opacus is not available. "
            "Install opacus (pip install opacus) or run without --dp."
        ) from e

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=1,  # we update epsilon tracking per-epoch externally; this is okay for demo
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, data_loader, DPEngine(privacy_engine, epsilon, delta)
