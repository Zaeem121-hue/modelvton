from fastapi import APIRouter, HTTPException
from app.schemas.tryon_schema import TryOnRequest, TryOnResponse, SizingCheckResult
from app.services.tryon_service import process_tryon
from app.services.sizing_service import (
    check_clothing_size,
    estimate_person_size,
    get_special_item_handling,
)
from app.schemas.tryon_schema import (
    BodyMeasurements,
    ClothingItemInput,
    StandardSize,
    SpecialItem,
)

router = APIRouter(prefix="/tryon", tags=["Virtual Try-On"])


@router.post("/process", response_model=TryOnResponse)
async def try_on(request: TryOnRequest):
    result = process_tryon(request)
    if result.status == "error":
        raise HTTPException(status_code=500, detail=result.error)
    if result.status == "size_mismatch":
        raise HTTPException(status_code=400, detail=result.message)
    return result


@router.post("/check-size", response_model=SizingCheckResult)
async def check_size(
    clothing: ClothingItemInput,
    person_measurements: BodyMeasurements,
):
    return check_clothing_size(clothing, person_measurements)


@router.post("/estimate-size")
async def estimate_size(measurements: BodyMeasurements):
    estimated, debug = estimate_person_size(measurements)
    return {
        "estimated_size": estimated.value if estimated else None,
        "debug": debug,
    }


@router.get("/special-item-info/{item_type}")
async def special_item_info(item_type: str):
    try:
        special = SpecialItem(item_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown special item: {item_type}")
    info = get_special_item_handling(ClothingItemInput(
        name=item_type,
        category="accessory",
        special_item=special,
    ))
    return info


@router.get("/sizes")
async def list_sizes():
    return {"sizes": [s.value for s in StandardSize]}


@router.get("/categories")
async def list_categories():
    from app.schemas.tryon_schema import ClothingCategory
    return {"categories": [c.value for c in ClothingCategory]}


@router.get("/special-items")
async def list_special_items():
    return {"special_items": [s.value for s in SpecialItem]}
