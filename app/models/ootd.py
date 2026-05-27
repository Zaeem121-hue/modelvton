import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple

from app.config import OOTD_MODEL_PATH, DEVICE, DTYPE


_ootd_pipeline = None


def load_ootd_model():
    global _ootd_pipeline
    if _ootd_pipeline is not None:
        return _ootd_pipeline

    try:
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = OOTD_MODEL_PATH
        if not os.path.exists(model_id):
            model_id = "levihsu/OOTDiffusion"

        dtype = torch.float16 if DTYPE == "fp16" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=OOTD_MODEL_PATH,
        )
        pipe = pipe.to(DEVICE)
        _ootd_pipeline = pipe
        print(f"OOTDiffusion model loaded on {DEVICE}")
        return _ootd_pipeline
    except Exception as e:
        print(f"Failed to load OOTDiffusion model: {e}")
        raise


def decode_base64(image_b64: str) -> Image.Image:
    image_data = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_data)).convert("RGB")


def encode_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_ootd_tryon(
    person_image_b64: str,
    clothing_image_b64: str,
    category: str,
    guidance_scale: float = 2.5,
    num_inference_steps: int = 30,
) -> Tuple[Image.Image, Optional[str]]:
    pipe = load_ootd_model()

    person_img = decode_base64(person_image_b64)
    cloth_img = decode_base64(clothing_image_b64)

    person_img_resized = person_img.resize((768, 1024))
    cloth_img_resized = cloth_img.resize((768, 1024))

    prompt = "a photo of a model wearing the outfit"
    negative_prompt = "low quality, distorted, ugly, deformed, blurry, bad anatomy"

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=person_img_resized,
        control_image=cloth_img_resized,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return result, None


def unload_ootd_model():
    global _ootd_pipeline
    if _ootd_pipeline is not None:
        import torch
        del _ootd_pipeline
        torch.cuda.empty_cache()
        _ootd_pipeline = None
