from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch

from .lanpaint import LanPaint as LanPaintEngine


class UnsupportedSchedulerError(RuntimeError):
    """Raised when a provided scheduler does not expose the required sigma/timestep API."""


def _unet_output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample

    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]

    raise TypeError("UNet output must be a Tensor or have a `.sample` Tensor attribute.")


def _broadcast_sigma(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return sigma.reshape([sigma.shape[0]] + [1] * (like.ndim - 1))


def _to_batch_tensor(value: torch.Tensor | float, *, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.to(device=device, dtype=dtype).expand(batch)
        if value.ndim == 1 and value.shape[0] == batch:
            return value.to(device=device, dtype=dtype)
        raise ValueError(f"Expected a scalar or (batch,) tensor, got shape {tuple(value.shape)}")

    return torch.full((batch,), float(value), device=device, dtype=dtype)


def _to_scalar_tensor(value: torch.Tensor | float, *, device: torch.device) -> torch.Tensor:
    """Convert to a device tensor scalar.

    Diffusers schedulers typically expect `timestep` to be a scalar (not a batch vector).
    """

    if isinstance(value, torch.Tensor):
        t = value.to(device=device)
        if t.ndim == 0:
            return t
        if t.numel() == 1:
            return t.reshape(())
        if t.ndim == 1:
            first = t[0]
            if not torch.all(t == first):
                raise ValueError("Expected a scalar timestep, or a (batch,) tensor with identical values.")
            return first.reshape(())
        raise ValueError(f"Expected a scalar or (batch,) timestep tensor, got shape {tuple(t.shape)}")

    return torch.tensor(value, device=device)


def _prepare_keep_mask(*, mask_inpaint: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    # Diffusers-style mask: 1 = hole/inpaint, 0 = keep.
    mask = (mask_inpaint > 0.5).to(device=latents.device, dtype=latents.dtype)

    if mask.ndim == latents.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != latents.ndim:
        raise ValueError(f"mask_inpaint must have ndim {latents.ndim - 1} or {latents.ndim}, got {mask.ndim}")

    if mask.shape[0] != latents.shape[0]:
        raise ValueError("mask_inpaint batch must match latents batch")

    if mask.shape[1] == 1 and latents.shape[1] != 1:
        mask = mask.repeat(1, latents.shape[1], *([1] * (latents.ndim - 2)))
    elif mask.shape[1] != latents.shape[1]:
        raise ValueError("mask_inpaint channel must be 1 or match latents channel")

    return 1.0 - mask


def _make_current_times_ve(*, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ve_sigma = sigma
    abt = 1.0 / (1.0 + ve_sigma**2)
    flow_t = ve_sigma / (1.0 + ve_sigma)
    return ve_sigma, abt, flow_t


class _ModelSamplingVE:
    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor) -> torch.Tensor:
        return latent_image + noise * sigma


@dataclass(frozen=True)
class _AdapterCfg:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor]
    guidance_scale: float
    guidance_scale_big: Optional[float]
    scale_model_input: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    unet_kwargs: Mapping[str, Any]


class _DiffusersEpsAdapter:
    """Adapter that turns an epsilon-predicting UNet into LanPaint's (x0, x0_big) callable."""

    def __init__(self, unet: Any, cfg: _AdapterCfg) -> None:
        self._unet = unet
        self._cfg = cfg

        # Mimic ComfyUI's nested model wrapper structure expected by `LanPaintEngine`.
        self.inner_model = self
        self.model_sampling = _ModelSamplingVE()

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        model_options: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep: torch.Tensor | float = sigma
        if isinstance(model_options, dict) and "timestep" in model_options:
            timestep = model_options["timestep"]

        t = _to_scalar_tensor(timestep, device=x.device)
        x_in = self._cfg.scale_model_input(x, t)

        big_scale = self._cfg.guidance_scale_big if self._cfg.guidance_scale_big is not None else self._cfg.guidance_scale
        needs_cfg = (self._cfg.guidance_scale != 1.0) or (big_scale != 1.0)

        eps_cond = _unet_output_to_tensor(
            self._unet(x_in, t, encoder_hidden_states=self._cfg.prompt_embeds, **dict(self._cfg.unet_kwargs))
        )

        if (not needs_cfg) or self._cfg.negative_prompt_embeds is None:
            eps_uncond = eps_cond
        else:
            eps_uncond = _unet_output_to_tensor(
                self._unet(x_in, t, encoder_hidden_states=self._cfg.negative_prompt_embeds, **dict(self._cfg.unet_kwargs))
            )

        eps_delta = eps_cond - eps_uncond
        eps = eps_uncond + self._cfg.guidance_scale * eps_delta
        eps_big = eps_uncond + big_scale * eps_delta

        sigma_view = _broadcast_sigma(sigma, x)
        x0 = x - sigma_view * eps
        x0_big = x - sigma_view * eps_big
        return x0, x0_big


def _step_output_prev_sample(step_output: Any) -> torch.Tensor:
    prev = getattr(step_output, "prev_sample", None)
    if isinstance(prev, torch.Tensor):
        return prev

    if isinstance(step_output, dict) and isinstance(step_output.get("prev_sample"), torch.Tensor):
        return step_output["prev_sample"]

    if isinstance(step_output, (tuple, list)) and step_output and isinstance(step_output[0], torch.Tensor):
        return step_output[0]

    raise TypeError("scheduler.step(...) must return an object/dict/tuple containing `prev_sample`.")


@torch.no_grad()
def lanpaint_diffusers_inpaint_latents(
    *,
    unet: Any,
    scheduler: Any,
    latents: torch.Tensor,
    latent_image: torch.Tensor,
    mask_inpaint: torch.Tensor,
    noise: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    guidance_scale: float,
    guidance_scale_big: Optional[float] = None,
    num_inference_steps: int,
    lanpaint_n_steps: int = 5,
    seed: Optional[int] = None,
    model_options: Optional[dict[str, Any]] = None,
    unet_kwargs: Optional[Mapping[str, Any]] = None,
    step_kwargs: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Run LanPaint with a diffusers-style epsilon UNet and sigma scheduler (duck-typed).

    Notes:
    - This integration intentionally avoids importing diffusers. The caller is responsible for
      providing a compatible `unet` and `scheduler` object.
    - `mask_inpaint` follows diffusers convention: 1 = hole/inpaint, 0 = keep.
    """

    if latents.shape != latent_image.shape:
        raise ValueError("latent_image must have the same shape as latents")
    if latents.shape != noise.shape:
        raise ValueError("noise must have the same shape as latents")

    keep_mask = _prepare_keep_mask(mask_inpaint=mask_inpaint, latents=latents)

    if hasattr(scheduler, "set_timesteps"):
        scheduler.set_timesteps(num_inference_steps, device=latents.device)

    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if timesteps is None or sigmas is None:
        raise UnsupportedSchedulerError("scheduler must expose `timesteps` and `sigmas`")

    timesteps_t = timesteps if isinstance(timesteps, torch.Tensor) else torch.tensor(list(timesteps), device=latents.device)
    sigmas_t = sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(list(sigmas), device=latents.device)

    if sigmas_t.numel() not in {timesteps_t.numel(), timesteps_t.numel() + 1}:
        raise UnsupportedSchedulerError("Expected len(sigmas) == len(timesteps) or len(timesteps)+1")

    scale_model_input = getattr(scheduler, "scale_model_input", None)
    if scale_model_input is None:
        scale_model_input = lambda x, t: x  # noqa: E731

    adapter = _DiffusersEpsAdapter(
        unet,
        _AdapterCfg(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            guidance_scale_big=guidance_scale_big,
            scale_model_input=scale_model_input,
            unet_kwargs={} if unet_kwargs is None else dict(unet_kwargs),
        ),
    )

    engine = LanPaintEngine(
        adapter,
        NSteps=lanpaint_n_steps,
        Friction=15.0,
        Lambda=4.0,
        Beta=1.0,
        StepSize=0.2,
        IS_FLUX=False,
        IS_FLOW=False,
    )

    base_model_options: dict[str, Any] = {} if model_options is None else dict(model_options)
    step_kwargs_dict: dict[str, Any] = {} if step_kwargs is None else dict(step_kwargs)

    n_steps_outer = timesteps_t.numel()
    for i in range(n_steps_outer):
        sigma_scalar = sigmas_t[i]
        t_scalar = timesteps_t[i]

        sigma = _to_batch_tensor(sigma_scalar, batch=latents.shape[0], device=latents.device, dtype=latents.dtype)
        current_times = _make_current_times_ve(sigma=sigma)

        opts = dict(base_model_options)
        opts["timestep"] = t_scalar

        x0 = engine(
            latents,
            latent_image,
            noise,
            sigma,
            keep_mask,
            current_times,
            opts,
            seed,
        )

        sigma_view = _broadcast_sigma(sigma, latents)
        eps = torch.where(sigma_view == 0, torch.zeros_like(latents), (latents - x0) / sigma_view)

        step_output = scheduler.step(eps, t_scalar, latents, **step_kwargs_dict)
        latents = _step_output_prev_sample(step_output)

    # Enforce mask at final sigma (usually 0) so the known region is exactly preserved.
    final_sigma = sigmas_t[n_steps_outer] if sigmas_t.numel() == n_steps_outer + 1 else torch.tensor(0.0, device=latents.device)
    final_sigma_b = _to_batch_tensor(final_sigma, batch=latents.shape[0], device=latents.device, dtype=latents.dtype)
    final_sigma_view = _broadcast_sigma(final_sigma_b, latents)
    latents = latents * (1.0 - keep_mask) + (latent_image + noise * final_sigma_view) * keep_mask

    return latents
