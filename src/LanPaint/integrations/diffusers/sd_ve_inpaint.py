from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...euler import euler_step_sigma
from ...lanpaint import LanPaint as LanPaintEngine
from ...masks import alpha_to_keep_mask, resize_keep_mask_to_latents
from ...timevars import make_current_times_ve

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from PIL.Image import Image as PILImage


def _require_diffusers():  # type: ignore[no-untyped-def]
    try:
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "diffusers is required for LanPaint's optional diffusers integration. "
            "Install it separately (e.g. `pip install diffusers transformers accelerate safetensors`)."
        ) from exc


def _require_pillow():  # type: ignore[no-untyped-def]
    try:
        from PIL import Image  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Pillow is required for image I/O. Install it with `pip install Pillow`.") from exc


@dataclass
class _VESampling:
    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor) -> torch.Tensor:
        return latent_image + noise * sigma


class _DiffusersX0VEModel:
    """Minimal k-diffusion style wrapper around a diffusers UNet (VE / Stable Diffusion)."""

    def __init__(
        self,
        *,
        unet: torch.nn.Module,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        guidance_scale_big: float,
    ) -> None:
        self.inner_model = self
        self.model_sampling = _VESampling()

        self._unet = unet
        self._prompt_embeds = prompt_embeds
        self._negative_prompt_embeds = negative_prompt_embeds
        self._guidance_scale = guidance_scale
        self._guidance_scale_big = guidance_scale_big

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        model_options: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if model_options is None or "timestep" not in model_options:
            raise ValueError("model_options['timestep'] is required for diffusers UNet conditioning")

        # Stable Diffusion (epsilon-pred) k-diffusion preconditioning:
        #   x_in = x / sqrt(sigma^2 + 1)
        sigma_b = sigma.reshape([sigma.shape[0]] + [1] * (x.ndim - 1))
        scale = torch.rsqrt(sigma_b.to(dtype=torch.float32) ** 2 + 1.0).to(dtype=x.dtype)
        x_in = x * scale

        timestep = model_options["timestep"]
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=x.device)
        timestep = timestep.to(device=x.device)

        # CFG in a single UNet forward.
        latent_in = torch.cat([x_in] * 2)
        if timestep.ndim == 0:
            t_in = timestep.expand(latent_in.shape[0])
        else:
            t_in = timestep

        emb = torch.cat([self._negative_prompt_embeds, self._prompt_embeds])
        noise_pred = self._unet(latent_in, t_in, encoder_hidden_states=emb, return_dict=False)[0]

        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_cfg = noise_uncond + self._guidance_scale * (noise_cond - noise_uncond)
        noise_cfg_big = noise_uncond + self._guidance_scale_big * (noise_cond - noise_uncond)

        x0 = x - sigma_b * noise_cfg
        x0_big = x - sigma_b * noise_cfg_big
        return x0, x0_big


def _gaussian_kernel(*, kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    if kernel_size == 1:
        return torch.ones((1, 1), device=device, dtype=dtype)

    sigma = (kernel_size - 1) / 4
    x = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    y = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _mask_blend(
    *, original: torch.Tensor, inpainted: torch.Tensor, inpaint_mask: torch.Tensor, blend_overlap: int
) -> torch.Tensor:
    """Blend `original` and `inpainted` with the ComfyUI LanPaint_MaskBlend logic.

    `inpaint_mask`: float in [0,1] shaped (1, H, W) where 1 means use inpainted.
    Images are float in [0,1] shaped (1, H, W, 3).
    """

    mask = inpaint_mask.float()
    mask = torch.nn.functional.max_pool2d(mask, kernel_size=blend_overlap, stride=1, padding=blend_overlap // 2)
    kernel = _gaussian_kernel(kernel_size=blend_overlap, device=mask.device, dtype=mask.dtype)[None, None, ...]
    mask = torch.nn.functional.conv2d(mask[:, None, :, :], kernel, padding=blend_overlap // 2)[:, 0, :, :]
    return original * (1 - mask[..., None]) + inpainted * mask[..., None]


def inpaint_sd_ve_lanpaint_euler_karras(  # noqa: PLR0913 (API surface)
    *,
    pipe: "StableDiffusionPipeline",
    init_image: "PILImage",
    mask_image: "PILImage",
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    steps: int = 30,
    cfg: float = 5.0,
    prompt_mode: str = "image_first",
    lanpaint_num_steps: int = 2,
    lanpaint_lambda: float = 4.0,
    lanpaint_step_size: float = 0.2,
    lanpaint_beta: float = 1.0,
    lanpaint_friction: float = 15.0,
    blend_overlap: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> "PILImage":
    """Training-free inpainting via LanPaint + diffusers Stable Diffusion (VE).

    Notes:
    - Sampler: Euler in sigma-space
    - Schedule: Karras sigmas (via diffusers EulerDiscreteScheduler)
    - Mask: ComfyUI-style from PNG alpha when present.
      Keep mask is `alpha/255` (1=keep, 0=inpaint), so inpaint mask is `1-keep`.
    """

    _require_diffusers()
    _require_pillow()
    from diffusers import EulerDiscreteScheduler
    from PIL import Image

    unet_param = next(pipe.unet.parameters())
    device: torch.device = unet_param.device
    dtype: torch.dtype = unet_param.dtype

    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("width and height must be provided together")
        init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
    else:
        width, height = init_image.size

    width = int(width)
    height = int(height)
    width8 = width - (width % 8)
    height8 = height - (height % 8)
    if (width8, height8) != (width, height):
        init_image = init_image.resize((width8, height8), resample=Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((width8, height8), resample=Image.Resampling.NEAREST)
        width, height = width8, height8

    # Encode init image to latents (deterministic: mode/mean).
    image_tensor = pipe.image_processor.preprocess(init_image).to(device=device, dtype=dtype)
    with torch.no_grad():
        latent_image = pipe.vae.encode(image_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor

    # Keep mask from alpha when available.
    mask_rgba = mask_image.convert("RGBA")
    alpha = torch.from_numpy(np.asarray(mask_rgba.getchannel("A")).copy()).to(device=device)
    keep_mask_px = alpha_to_keep_mask(alpha)
    keep_mask_lat = resize_keep_mask_to_latents(keep_mask_px=keep_mask_px, latents=latent_image)

    # Prompt embeddings (CFG always on; LanPaint uses x0_big for PromptMode).
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

    guidance_scale_big = cfg if prompt_mode.lower() == "image_first" else -0.5

    model = _DiffusersX0VEModel(
        unet=pipe.unet,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=cfg,
        guidance_scale_big=guidance_scale_big,
    )

    engine = LanPaintEngine(
        model,
        NSteps=lanpaint_num_steps,
        Friction=lanpaint_friction,
        Lambda=lanpaint_lambda,
        Beta=lanpaint_beta,
        StepSize=lanpaint_step_size,
        IS_FLUX=False,
        IS_FLOW=False,
    )

    # Scheduler: Euler + Karras sigmas.
    scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    scheduler.set_timesteps(steps, device=device)
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    timesteps = scheduler.timesteps.to(device=device)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latent_image.shape, generator=generator, device=device, dtype=dtype)

    x = latent_image + noise * sigmas[0]

    with torch.no_grad():
        for i in range(steps):
            sigma = sigmas[i].reshape(1)
            current_times = make_current_times_ve(sigma=sigma)
            model_options = {"timestep": timesteps[i]}

            x0 = engine(
                x,
                latent_image,
                noise,
                sigma,
                keep_mask_lat,
                current_times,
                model_options=model_options,
                seed=seed,
            )
            x = euler_step_sigma(x=x, x0=x0, sigma_from=sigmas[i], sigma_to=sigmas[i + 1])

        # Decode.
        x = x / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(x, return_dict=False)[0]
        pil_out = pipe.image_processor.postprocess(image, output_type="pil")[0]

    if blend_overlap is None:
        return pil_out

    # Optional: blend like LanPaint_MaskBlend (inpaint mask is 1 - keep mask).
    inpaint_mask_px = (1.0 - keep_mask_px).to(device=device, dtype=torch.float32)[None, ...]
    original = torch.from_numpy(np.asarray(init_image).astype("float32") / 255.0).to(device=device)
    inpainted = torch.from_numpy(np.asarray(pil_out).astype("float32") / 255.0).to(device=device)

    blended = _mask_blend(
        original=original[None, ...],
        inpainted=inpainted[None, ...],
        inpaint_mask=inpaint_mask_px,
        blend_overlap=int(blend_overlap),
    )
    blended = (blended.clamp(0, 1) * 255.0).to(dtype=torch.uint8)[0].cpu().numpy()
    return Image.fromarray(blended)
