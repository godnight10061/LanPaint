import torch


def make_current_times_ve(*, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LanPaint time variables from a VE sigma.

    Args:
        sigma: VE sigma (the usual ComfyUI meaning of `sigma` for non-Flow models).

    Returns:
        (ve_sigma, abt, flow_t) where `abt = 1 / (1 + ve_sigma^2)` and `flow_t = ve_sigma / (1 + ve_sigma)`.
    """
    ve_sigma = sigma
    abt = 1.0 / (1.0 + ve_sigma**2)
    flow_t = ve_sigma / (1.0 + ve_sigma)
    return ve_sigma, abt, flow_t


def make_current_times_flow(*, flow_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LanPaint time variables from a Flow/Flux `flow_t`.

    Args:
        flow_t: Flow time in (0, 1) (ComfyUI's `sigma` for Flow/Flux-style models).

    Returns:
        (ve_sigma, abt, flow_t) where `abt = (1 - flow_t)^2 / ((1 - flow_t)^2 + flow_t^2)` and
        `ve_sigma = flow_t / (1 - flow_t)` (with a small clamp for stability).
    """
    one_minus_flow_t = 1.0 - flow_t
    abt = one_minus_flow_t**2 / (one_minus_flow_t**2 + flow_t**2)
    ve_sigma = flow_t / torch.clamp(one_minus_flow_t, min=1e-9)
    return ve_sigma, abt, flow_t
