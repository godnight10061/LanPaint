from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.LanPaint.diffusers import lanpaint_diffusers_inpaint_latents


@dataclass
class _StepOutput:
    prev_sample: torch.Tensor


class _DummySigmaScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.sigmas = torch.tensor([], dtype=torch.float32)
        self._step_index = 0

    def set_timesteps(self, num_inference_steps: int, device: torch.device | None = None) -> None:
        # Fixed schedule: 2 outer steps (3 sigmas).
        if num_inference_steps != 2:
            raise ValueError("Dummy scheduler only supports num_inference_steps=2")
        self.timesteps = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device)
        self.sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32, device=device)
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return sample

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, **_: Any) -> _StepOutput:  # noqa: ARG002
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        prev_sample = sample + (sigma_next - sigma) * model_output
        return _StepOutput(prev_sample=prev_sample)


class _DummyUNet:
    def __init__(self) -> None:
        self.seen_timesteps: list[torch.Tensor] = []

    def __call__(self, sample: torch.Tensor, timestep: torch.Tensor, *, encoder_hidden_states: torch.Tensor, **_: Any) -> Any:
        self.seen_timesteps.append(timestep.detach().clone())
        # Deterministic epsilon: differs for cond/uncond via encoder_hidden_states.
        eps = torch.zeros_like(sample) + encoder_hidden_states.mean().to(sample.dtype) * 0.01
        return SimpleNamespace(sample=eps)


def test_lanpaint_diffusers_inpaint_latents_preserves_known_and_cfg_affects_hole() -> None:
    torch.manual_seed(0)
    unet = _DummyUNet()
    scheduler = _DummySigmaScheduler()

    latents = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    latent_image = torch.ones_like(latents)
    noise = torch.randn_like(latents)

    # Diffusers-style mask: 1 = inpaint/hole, 0 = keep.
    mask_inpaint = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mask_inpaint[:, :, 2:6, 2:6] = 1.0

    prompt_embeds = torch.ones((1, 4), dtype=torch.float32)
    negative_prompt_embeds = torch.zeros((1, 4), dtype=torch.float32)

    torch.manual_seed(0)
    out_cfg1 = lanpaint_diffusers_inpaint_latents(
        unet=unet,
        scheduler=scheduler,
        latents=latents,
        latent_image=latent_image,
        mask_inpaint=mask_inpaint,
        noise=noise,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=1.0,
        guidance_scale_big=1.0,
        num_inference_steps=2,
        lanpaint_n_steps=1,
        seed=0,
    )

    # Reset and run with higher CFG (should change only the hole region).
    torch.manual_seed(0)
    unet2 = _DummyUNet()
    scheduler2 = _DummySigmaScheduler()
    out_cfg2 = lanpaint_diffusers_inpaint_latents(
        unet=unet2,
        scheduler=scheduler2,
        latents=latents,
        latent_image=latent_image,
        mask_inpaint=mask_inpaint,
        noise=noise,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=2.0,
        guidance_scale_big=2.0,
        num_inference_steps=2,
        lanpaint_n_steps=1,
        seed=0,
    )

    keep_mask = (1.0 - mask_inpaint).repeat(1, latents.shape[1], 1, 1)
    hole_mask = mask_inpaint.repeat(1, latents.shape[1], 1, 1)

    assert out_cfg1.shape == latents.shape
    assert out_cfg2.shape == latents.shape
    assert torch.equal(out_cfg1[keep_mask == 1.0], latent_image[keep_mask == 1.0])
    assert torch.equal(out_cfg2[keep_mask == 1.0], latent_image[keep_mask == 1.0])
    assert not torch.allclose(out_cfg1[hole_mask == 1.0], out_cfg2[hole_mask == 1.0])

    # Ensure we passed scheduler timesteps (not sigmas) through to the UNet.
    seen = {float(t.mean().item()) for t in unet2.seen_timesteps}
    assert seen == {1.0, 2.0}


def test_lanpaint_diffusers_inpaint_latents_negative_prompt_none_disables_cfg() -> None:
    torch.manual_seed(0)
    unet = _DummyUNet()
    scheduler = _DummySigmaScheduler()

    latents = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    latent_image = torch.ones_like(latents)
    noise = torch.randn_like(latents)

    mask_inpaint = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    mask_inpaint[:, :, 2:6, 2:6] = 1.0

    prompt_embeds = torch.ones((1, 4), dtype=torch.float32)

    out_cfg1 = lanpaint_diffusers_inpaint_latents(
        unet=unet,
        scheduler=scheduler,
        latents=latents,
        latent_image=latent_image,
        mask_inpaint=mask_inpaint,
        noise=noise,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_scale=1.0,
        guidance_scale_big=1.0,
        num_inference_steps=2,
        lanpaint_n_steps=0,
        seed=0,
    )

    out_cfg2 = lanpaint_diffusers_inpaint_latents(
        unet=_DummyUNet(),
        scheduler=_DummySigmaScheduler(),
        latents=latents,
        latent_image=latent_image,
        mask_inpaint=mask_inpaint,
        noise=noise,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_scale=2.0,
        guidance_scale_big=2.0,
        num_inference_steps=2,
        lanpaint_n_steps=0,
        seed=0,
    )

    assert torch.allclose(out_cfg1, out_cfg2)


def test_lanpaint_diffusers_inpaint_latents_euler_scheduler_batch_uses_scalar_timestep() -> None:
    diffusers = pytest.importorskip("diffusers")
    EulerDiscreteScheduler = diffusers.EulerDiscreteScheduler  # noqa: N806

    torch.manual_seed(0)
    unet = _DummyUNet()
    scheduler = EulerDiscreteScheduler(num_train_timesteps=100)

    batch = 2
    latents = torch.zeros((batch, 4, 8, 8), dtype=torch.float32)
    latent_image = torch.ones_like(latents)
    noise = torch.randn_like(latents)

    mask_inpaint = torch.zeros((batch, 1, 8, 8), dtype=torch.float32)
    mask_inpaint[:, :, 2:6, 2:6] = 1.0

    prompt_embeds = torch.ones((batch, 4), dtype=torch.float32)
    negative_prompt_embeds = torch.zeros((batch, 4), dtype=torch.float32)

    out = lanpaint_diffusers_inpaint_latents(
        unet=unet,
        scheduler=scheduler,
        latents=latents,
        latent_image=latent_image,
        mask_inpaint=mask_inpaint,
        noise=noise,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=2.0,
        guidance_scale_big=2.0,
        num_inference_steps=1,
        lanpaint_n_steps=0,
        seed=0,
    )

    keep_mask = (1.0 - mask_inpaint).repeat(1, latents.shape[1], 1, 1)
    assert out.shape == latents.shape
    assert torch.equal(out[keep_mask == 1.0], latent_image[keep_mask == 1.0])


def test_lanpaint_diffusers_inpaint_latents_is_no_grad() -> None:
    unet = _DummyUNet()
    scheduler = _DummySigmaScheduler()

    latents = torch.zeros((1, 4, 8, 8), dtype=torch.float32, requires_grad=True)
    latent_image = torch.ones_like(latents)
    noise = torch.randn_like(latents)
    mask_inpaint = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

    prompt_embeds = torch.ones((1, 4), dtype=torch.float32)
    negative_prompt_embeds = torch.zeros((1, 4), dtype=torch.float32)

    out = lanpaint_diffusers_inpaint_latents(
        unet=unet,
        scheduler=scheduler,
        latents=latents,
        latent_image=latent_image,
        mask_inpaint=mask_inpaint,
        noise=noise,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=1.0,
        guidance_scale_big=1.0,
        num_inference_steps=2,
        lanpaint_n_steps=0,
        seed=0,
    )

    assert not out.requires_grad

