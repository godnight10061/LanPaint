"""Diffusers integration (optional).

This module is intentionally not imported by default to keep LanPaint usable as a
ComfyUI extension without requiring `diffusers` to be installed.
"""

from .sd_ve_inpaint import inpaint_sd_ve_lanpaint_euler_karras

__all__ = ["inpaint_sd_ve_lanpaint_euler_karras"]

