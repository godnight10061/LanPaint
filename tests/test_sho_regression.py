import torch
from torch.distributions import Distribution

from src.LanPaint.utils import StochasticHarmonicOscillator


def test_dynamics_does_not_pass_nan_to_multivariatenormal() -> None:
    torch.manual_seed(0)

    prev_validate_args = Distribution._validate_args
    Distribution.set_default_validate_args(True)
    try:
        osc = StochasticHarmonicOscillator(
            Gamma=torch.tensor(0.0),
            A=torch.tensor(1.0),
            C=torch.tensor(0.0),
            D=torch.tensor(1.0),
        )

        y1, v1 = osc.dynamics(
            y0=torch.zeros((1,)),
            v0=torch.zeros((1,)),
            t=torch.tensor(0.1),
        )
    finally:
        Distribution.set_default_validate_args(prev_validate_args)

    assert torch.isfinite(y1).all()
    assert torch.isfinite(v1).all()
