import os
import uuid

from app.schemas.tryon_schema import (
    TryOnRequest,
    TryOnResponse,
    SpecialItem,
)
from app.services.sizing_service import (
    check_clothing_size,
    get_special_item_handling,
)
from app.models.ootd import run_ootd_tryon, encode_base64
from app.models.omnitry import run_omnitry_tryon, encode_b64 as omnitry_encode
from app.config import UPLOAD_DIR


def process_tryon(request: TryOnRequest) -> TryOnResponse:
    sizing_result = check_clothing_size(request.clothing, request.person_measurements)

    if not sizing_result.passed:
        return TryOnResponse(
            status="size_mismatch",
            sizing=sizing_result,
            message=sizing_result.message,
        )

    special_info = get_special_item_handling(request.clothing)
    override = special_info.get("override_size_check", False)

    if request.clothing.special_item != SpecialItem.NONE:
        sizing_result.message = (
            f"Special item '{request.clothing.special_item.value}' detected. "
            f"{special_info['sizing_note']}"
        )
        if not override and request.person_measurements:
            sizing_override = check_clothing_size(request.clothing, request.person_measurements)
            if not sizing_override.passed:
                return TryOnResponse(
                    status="size_mismatch",
                    sizing=sizing_override,
                    message=sizing_override.message,
                )

    try:
        if request.model_type == "omnitry":
            result_img, warning = run_omnitry_tryon(
                person_image_b64=request.person_image,
                clothing_image_b64=request.clothing_image,
                category=request.clothing.category.value,
            )
            result_b64 = omnitry_encode(result_img)
        else:
            result_img, warning = run_ootd_tryon(
                person_image_b64=request.person_image,
                clothing_image_b64=request.clothing_image,
                category=request.clothing.category.value,
            )
            result_b64 = encode_base64(result_img)

        save_path = os.path.join(UPLOAD_DIR, f"tryon_{uuid.uuid4().hex}.png")
        result_img.save(save_path)

        return TryOnResponse(
            status="success",
            result_image=result_b64,
            sizing=sizing_result,
            message="Try-on completed successfully.",
        )

    except Exception as e:
        return TryOnResponse(
            status="error",
            error=str(e),
            sizing=sizing_result,
            message="Try-on processing failed.",
        )
