import torch


def euler_step_sigma(*, x: torch.Tensor, x0: torch.Tensor, sigma_from: torch.Tensor, sigma_to: torch.Tensor) -> torch.Tensor:
    """One Euler step in sigma-space using an x0 (predicted original sample).

    This matches the k-diffusion `sample_euler` update:
        d = (x - x0) / sigma
        x_next = x + d * (sigma_next - sigma)
    """

    if sigma_from.ndim != 0:
        raise ValueError("sigma_from must be a scalar tensor")
    if sigma_to.ndim != 0:
        raise ValueError("sigma_to must be a scalar tensor")

    if torch.allclose(sigma_from, sigma_to):
        return x
    if torch.allclose(sigma_from, x.new_zeros(())):
        return x0

    d = (x - x0) / sigma_from
    return x + d * (sigma_to - sigma_from)
