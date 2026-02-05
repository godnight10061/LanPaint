import torch


def make_current_times_ve(*, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ve_sigma = sigma
    abt = 1.0 / (1.0 + ve_sigma**2)
    sqrt_1_minus_abt = torch.sqrt(1.0 - abt)
    flow_t = sqrt_1_minus_abt / (sqrt_1_minus_abt + torch.sqrt(abt))
    return ve_sigma, abt, flow_t


def make_current_times_flow(*, flow_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    one_minus_flow_t = 1.0 - flow_t
    abt = one_minus_flow_t**2 / (one_minus_flow_t**2 + flow_t**2)
    ve_sigma = flow_t / torch.clamp(one_minus_flow_t, min=1e-9)
    return ve_sigma, abt, flow_t

