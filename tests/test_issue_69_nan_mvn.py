import warnings

import torch
from torch.distributions import Distribution

from src.LanPaint.lanpaint import LanPaint


def test_issue_69_fp16_coefficients_do_not_generate_nan_loc() -> None:
    """Regression for issue 69: fp16 coefficient overflow produced NaNs in MVN loc.

    This reproduces the original failure mode without injecting NaNs directly:
    with fp16 inputs, the internal Gamma coefficient can overflow to inf when dt is small,
    which then propagates NaNs into the MultivariateNormal mean (loc) under validate_args=True.
    """

    torch.manual_seed(0)

    warnings.filterwarnings(
        "ignore",
        message=r"In CPU autocast, but the target dtype is not supported\.",
        category=UserWarning,
    )

    prev_validate_args = Distribution._validate_args
    Distribution.set_default_validate_args(True)
    try:
        lanpaint = LanPaint(
            Model=None,
            NSteps=1,
            Friction=15.0,
            Lambda=16.0,
            Beta=1.0,
            StepSize=0.2,
        )

        x_t = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float16)
        lanpaint.img_dim_size = x_t.ndim

        mask = torch.zeros_like(x_t)
        batch = x_t.shape[0]

        abt = torch.full((batch,), 0.99, dtype=x_t.dtype)
        current_times = (
            torch.full((batch,), 0.1, dtype=x_t.dtype),  # sigma (unused by score in this test)
            abt,
            torch.zeros((batch,), dtype=x_t.dtype),
        )

        step_size = lanpaint.add_none_dims(lanpaint.step_size * (1 - abt))

        def score(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        out, _ = lanpaint.langevin_dynamics(
            x_t=x_t,
            score=score,
            mask=mask,
            step_size=step_size,
            current_times=current_times,
            sigma_x=lanpaint.add_none_dims(lanpaint.sigma_x(abt)),
            sigma_y=lanpaint.add_none_dims(lanpaint.sigma_y(abt)),
            args=None,
        )
    finally:
        Distribution.set_default_validate_args(prev_validate_args)

    assert torch.isfinite(out).all()
