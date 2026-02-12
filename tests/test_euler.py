import torch

from src.LanPaint.euler import euler_step_sigma


def test_euler_step_sigma_identity_when_sigma_unchanged() -> None:
    x = torch.randn(1, 4, 2, 2)
    x0 = torch.randn_like(x)
    sigma = torch.tensor(1.0)
    out = euler_step_sigma(x=x, x0=x0, sigma_from=sigma, sigma_to=sigma)
    assert torch.equal(out, x)


def test_euler_step_sigma_hits_x0_at_zero_sigma() -> None:
    x = torch.randn(1, 4, 2, 2)
    x0 = torch.randn_like(x)
    out = euler_step_sigma(x=x, x0=x0, sigma_from=torch.tensor(1.0), sigma_to=torch.tensor(0.0))
    assert torch.allclose(out, x0)

