import torch

from src.LanPaint.lanpaint import LanPaint as LanPaintEngine


class _DummySampling:
    def noise_scaling(self, sigma, noise, latent_image):  # type: ignore[no-untyped-def]
        return latent_image + noise * sigma


class _DummyModel:
    def __init__(self) -> None:
        self.inner_model = self
        self.model_sampling = _DummySampling()
        self.seen_times: list[torch.Tensor] = []

    def __call__(self, x, sigma, model_options=None, seed=None):  # type: ignore[no-untyped-def]
        self.seen_times.append(sigma.detach().clone())
        return x, x


def _make_current_times_ve(*, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ve_sigma = sigma
    abt = 1.0 / (1.0 + ve_sigma**2)
    flow_t = torch.sqrt(1.0 - abt) / (torch.sqrt(1.0 - abt) + torch.sqrt(abt))
    return ve_sigma, abt, flow_t


def _make_current_times_flow(*, flow_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abt = (1.0 - flow_t) ** 2 / ((1.0 - flow_t) ** 2 + flow_t**2)
    ve_sigma = flow_t / (1.0 - flow_t)
    return ve_sigma, abt, flow_t


def test_pure_python_usage_smoke_ve() -> None:
    torch.manual_seed(0)
    model = _DummyModel()
    engine = LanPaintEngine(
        model,
        NSteps=2,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
        IS_FLUX=False,
        IS_FLOW=False,
    )

    x = torch.zeros((1, 4, 8, 8))
    latent_image = torch.ones_like(x)
    noise = torch.randn_like(x)
    sigma = torch.tensor([1.0])

    latent_mask = torch.ones_like(x)
    latent_mask[:, :, 2:6, 2:6] = 0.0

    current_times = _make_current_times_ve(sigma=sigma)

    out = engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options={},
        seed=0,
        n_steps=2,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out[latent_mask == 1.0], latent_image[latent_mask == 1.0])
    assert model.seen_times
    assert torch.allclose(model.seen_times[-1].float().mean(), sigma.float().mean())


def test_pure_python_usage_smoke_flux_flow() -> None:
    torch.manual_seed(0)
    model = _DummyModel()
    engine = LanPaintEngine(
        model,
        NSteps=2,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
        IS_FLUX=True,
        IS_FLOW=False,
    )

    x = torch.zeros((1, 4, 8, 8))
    latent_image = torch.ones_like(x)
    noise = torch.randn_like(x)
    sigma = torch.tensor([0.5])  # flow time

    latent_mask = torch.ones_like(x)
    latent_mask[:, :, 2:6, 2:6] = 0.0

    current_times = _make_current_times_flow(flow_t=sigma)

    out = engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options={},
        seed=0,
        n_steps=2,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out[latent_mask == 1.0], latent_image[latent_mask == 1.0])
    assert model.seen_times
    assert torch.allclose(model.seen_times[-1].float().mean(), sigma.float().mean())
