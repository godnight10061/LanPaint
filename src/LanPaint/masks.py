from __future__ import annotations

import torch


def alpha_to_keep_mask(alpha: torch.Tensor) -> torch.Tensor:
    """Convert an alpha channel to a keep-mask.

    Semantics follow LanPaint core:
      - `1.0` means known / keep
      - `0.0` means to inpaint

    Accepts alpha in either `[0, 1]` float or `[0, 255]` integer/float.
    """

    alpha_f = alpha.to(dtype=torch.float32)
    if alpha_f.numel() == 0:
        raise ValueError("alpha must be non-empty")

    if alpha_f.max() > 1.0:
        alpha_f = alpha_f / 255.0

    return alpha_f.clamp(0.0, 1.0)


def resize_keep_mask_to_latents(*, keep_mask_px: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Resize a pixel-space keep mask to a latent-space mask, using nearest sampling.

    Returns a tensor shaped like `latents` (B, C, H, W) with float values in [0, 1].
    """

    if keep_mask_px.ndim == 2:
        keep_mask_px = keep_mask_px[None, None, ...]
    elif keep_mask_px.ndim == 3:
        keep_mask_px = keep_mask_px[:, None, ...]
    elif keep_mask_px.ndim != 4:
        raise ValueError("keep_mask_px must have shape (H,W), (B,H,W) or (B,1,H,W)")

    if latents.ndim != 4:
        raise ValueError("latents must have shape (B,C,H,W)")

    target_hw = (latents.shape[-2], latents.shape[-1])
    mask_lat = torch.nn.functional.interpolate(keep_mask_px, size=target_hw, mode="nearest")
    if mask_lat.shape[1] == 1 and latents.shape[1] != 1:
        mask_lat = mask_lat.repeat(1, latents.shape[1], 1, 1)
    return mask_lat.to(device=latents.device, dtype=latents.dtype)
