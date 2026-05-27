import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple

from app.config import OMNITRY_MODEL_PATH, DEVICE, DTYPE


_omnitry_pipeline = None


def load_omnitry_model():
    global _omnitry_pipeline
    if _omnitry_pipeline is not None:
        return _omnitry_pipeline

    try:
        from diffusers import StableDiffusionInpaintPipeline
        import torch

        dtype = torch.float16 if DTYPE == "fp16" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
        )
        pipe = pipe.to(DEVICE)
        _omnitry_pipeline = pipe
        print(f"OmniTry model loaded on {DEVICE}")
        return _omnitry_pipeline
    except Exception as e:
        print(f"OmniTry model not available, using fallback: {e}")
        return None


def decode_b64(image_b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")


def encode_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_inpaint_mask(person_img: Image.Image, cloth_img: Image.Image) -> Image.Image:
    person_arr = np.array(person_img.resize((512, 512)))
    cloth_arr = np.array(cloth_img.resize((512, 512)))

    gray_cloth = np.mean(cloth_arr, axis=2)
    cloth_mask = (gray_cloth < 250).astype(np.uint8) * 255

    mask = Image.fromarray(cloth_mask).resize(person_img.size)
    return mask


def run_omnitry_tryon(
    person_image_b64: str,
    clothing_image_b64: str,
    category: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> Tuple[Image.Image, Optional[str]]:
    pipe = load_omnitry_model()
    if pipe is None:
        person_img = decode_b64(person_image_b64)
        cloth_img = decode_b64(clothing_image_b64)
        cloth_resized = cloth_img.resize(person_img.size)
        blended = Image.blend(person_img, cloth_resized, alpha=0.3)
        return blended, "OmniTry model unavailable; used fallback blend."

    person_img = decode_b64(person_image_b64)
    cloth_img = decode_b64(clothing_image_b64)

    person_resized = person_img.resize((512, 512))
    cloth_resized = cloth_img.resize((512, 512))
    mask = create_inpaint_mask(person_resized, cloth_resized)

    prompt = "a photo of a person wearing the clothing item, photorealistic, high quality"
    negative_prompt = "low quality, distorted, ugly"

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=person_resized,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    return result, None


def unload_omnitry_model():
    global _omnitry_pipeline
    if _omnitry_pipeline is not None:
        import torch
        del _omnitry_pipeline
        torch.cuda.empty_cache()
        _omnitry_pipeline = None
