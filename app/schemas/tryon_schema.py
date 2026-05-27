from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ClothingCategory(str, Enum):
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    DRESS = "dress"
    OUTERWEAR = "outerwear"
    ACCESSORY = "accessory"


class SpecialItem(str, Enum):
    TIE = "tie"
    BELT = "belt"
    SCARF = "scarf"
    HAT = "hat"
    GLOVES = "gloves"
    SUSPENDERS = "suspenders"
    NONE = "none"


class StandardSize(str, Enum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"
    ONE_SIZE = "one_size"


class BodyMeasurements(BaseModel):
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hip_cm: Optional[float] = None
    shoulder_cm: Optional[float] = None
    neck_cm: Optional[float] = None
    arm_length_cm: Optional[float] = None
    inseam_cm: Optional[float] = None


class ClothingItemInput(BaseModel):
    name: str
    category: ClothingCategory
    size: Optional[StandardSize] = None
    measurements: Optional[BodyMeasurements] = None
    special_item: SpecialItem = SpecialItem.NONE


class TryOnRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    person_image: str = Field(..., description="Base64 encoded person image")
    clothing_image: str = Field(..., description="Base64 encoded clothing image")
    clothing: ClothingItemInput
    person_measurements: Optional[BodyMeasurements] = None
    model_type: str = Field(default="ootd", description="'ootd' or 'omnitry'")


class SizingCheckResult(BaseModel):
    passed: bool
    message: Optional[str] = None
    estimated_person_size: Optional[str] = None
    clothing_size: Optional[str] = None
    mismatches: List[str] = []


class TryOnResponse(BaseModel):
    status: str
    result_image: Optional[str] = None
    sizing: Optional[SizingCheckResult] = None
    message: Optional[str] = None
    error: Optional[str] = None
