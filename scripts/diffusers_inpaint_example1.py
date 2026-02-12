import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.LanPaint.integrations.diffusers import inpaint_sd_ve_lanpaint_euler_karras


def main() -> None:
    model_id = os.environ.get("LANPAINT_MODEL", "runwayml/stable-diffusion-v1-5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

    init_image = Image.open(ROOT_DIR / "examples/Example_1/Original_No_Mask.png").convert("RGB")
    mask_image = Image.open(ROOT_DIR / "examples/Example_1/Masked_Load_Me_in_Loader.png").convert("RGBA")

    prompt = "basketball, masterpiece, high score, great score, absurdres"
    negative = (
        "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, "
        "worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    )

    out = inpaint_sd_ve_lanpaint_euler_karras(
        pipe=pipe,
        init_image=init_image,
        mask_image=mask_image,
        prompt=prompt,
        negative_prompt=negative,
        seed=0,
        steps=30,
        cfg=5.0,
        prompt_mode="image_first",
        lanpaint_num_steps=2,
        lanpaint_lambda=4.0,
        lanpaint_step_size=0.2,
        lanpaint_beta=1.0,
        lanpaint_friction=15.0,
        blend_overlap=15,
        width=512,
        height=720,
    )

    out_dir = ROOT_DIR / "output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "diffusers_lanpaint_example1_sd15.png"
    out.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
