import torch
import pytest
from typing import Optional

from src.LanPaint.lanpaint import LanPaint as LanPaintEngine
from src.LanPaint.timevars import make_current_times_flow, make_current_times_ve


class _DummySampling:
    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor) -> torch.Tensor:
        return latent_image + noise * sigma


class _DummyModel:
    def __init__(self) -> None:
        self.inner_model = self
        self.model_sampling = _DummySampling()
        self.seen_times: list[torch.Tensor] = []

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        model_options: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_times.append(sigma.detach().clone())
        return x, x


@pytest.mark.parametrize(
    ("is_flux", "is_flow", "sigma_val"),
    [
        (False, False, 1.0),
        (True, False, 0.5),
        (False, True, 0.5),
    ],
    ids=["ve", "flux", "flow"],
)
def test_pure_python_usage_smoke(*, is_flux: bool, is_flow: bool, sigma_val: float) -> None:
    torch.manual_seed(0)
    model = _DummyModel()
    n_inner_steps = 2
    engine = LanPaintEngine(
        model,
        NSteps=n_inner_steps,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
        IS_FLUX=is_flux,
        IS_FLOW=is_flow,
    )

    x = torch.zeros((1, 4, 8, 8))
    latent_image = torch.ones_like(x)
    noise = torch.randn_like(x)
    sigma = torch.tensor([sigma_val])  # VE sigma or flow_t (Flux/Flow)

    latent_mask = torch.ones_like(x)
    latent_mask[:, :, 2:6, 2:6] = 0.0

    if is_flux or is_flow:
        current_times = make_current_times_flow(flow_t=sigma)
    else:
        current_times = make_current_times_ve(sigma=sigma)

    out = engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options={},
        seed=0,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out[latent_mask == 1.0], latent_image[latent_mask == 1.0])
    assert len(model.seen_times) == n_inner_steps + 1
    assert all(torch.allclose(t, sigma) for t in model.seen_times)
