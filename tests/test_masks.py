import torch

from src.LanPaint.masks import alpha_to_keep_mask, resize_keep_mask_to_latents


def test_alpha_to_keep_mask_uint8_255_scale() -> None:
    alpha = torch.tensor([[0, 255]], dtype=torch.uint8)
    keep = alpha_to_keep_mask(alpha)
    assert keep.dtype == torch.float32
    assert torch.allclose(keep, torch.tensor([[0.0, 1.0]]))


def test_resize_keep_mask_to_latents_nearest_and_broadcast() -> None:
    keep_px = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    latents = torch.zeros((1, 4, 2, 2))
    keep_lat = resize_keep_mask_to_latents(keep_mask_px=keep_px, latents=latents)
    assert keep_lat.shape == latents.shape
    expected = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert torch.allclose(keep_lat[:, :1], expected)
    assert torch.allclose(keep_lat[:, 1:], expected.repeat(1, 3, 1, 1))

