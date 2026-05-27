from typing import Optional, Tuple, List
from app.schemas.tryon_schema import (
    BodyMeasurements,
    ClothingItemInput,
    ClothingCategory,
    SpecialItem,
    StandardSize,
    SizingCheckResult,
)


SIZE_CHART = {
    StandardSize.XS: {
        "chest_cm": (78, 86),
        "waist_cm": (60, 68),
        "hip_cm": (84, 92),
        "shoulder_cm": (36, 38),
        "neck_cm": (33, 35),
    },
    StandardSize.S: {
        "chest_cm": (86, 94),
        "waist_cm": (68, 76),
        "hip_cm": (92, 100),
        "shoulder_cm": (38, 40),
        "neck_cm": (35, 37),
    },
    StandardSize.M: {
        "chest_cm": (94, 102),
        "waist_cm": (76, 84),
        "hip_cm": (100, 108),
        "shoulder_cm": (40, 42),
        "neck_cm": (37, 39),
    },
    StandardSize.L: {
        "chest_cm": (102, 110),
        "waist_cm": (84, 92),
        "hip_cm": (108, 116),
        "shoulder_cm": (42, 44),
        "neck_cm": (39, 41),
    },
    StandardSize.XL: {
        "chest_cm": (110, 118),
        "waist_cm": (92, 100),
        "hip_cm": (116, 124),
        "shoulder_cm": (44, 46),
        "neck_cm": (41, 43),
    },
    StandardSize.XXL: {
        "chest_cm": (118, 128),
        "waist_cm": (100, 110),
        "hip_cm": (124, 134),
        "shoulder_cm": (46, 49),
        "neck_cm": (43, 46),
    },
}


SPECIAL_ITEM_SIZE_RULES = {
    SpecialItem.TIE: "length-based; standard length ~145cm fits most adults",
    SpecialItem.BELT: "adjustable; measured from buckle to middle hole",
    SpecialItem.SCARF: "one-size; length and width are style choices",
    SpecialItem.HAT: "head circumference in cm is the key measure",
    SpecialItem.GLOVES: "hand circumference and finger length",
    SpecialItem.SUSPENDERS: "adjustable strap length; fits most adults",
    SpecialItem.NONE: "standard sizing rules apply",
}


SPECIAL_ITEMS_BOUGHT_WITH_OUTFIT = {
    SpecialItem.TIE,
    SpecialItem.BELT,
    SpecialItem.SCARF,
    SpecialItem.SUSPENDERS,
}


def estimate_person_size(measurements: BodyMeasurements) -> Tuple[Optional[StandardSize], List[str]]:
    debug_msgs = []
    if not any([
        measurements.chest_cm,
        measurements.waist_cm,
        measurements.hip_cm,
        measurements.shoulder_cm,
    ]):
        return None, ["No body measurements provided to estimate size"]

    scores = {s: 0 for s in StandardSize if s != StandardSize.ONE_SIZE}
    total_checks = 0

    for size_key, ranges in SIZE_CHART.items():
        score = 0
        checks = 0

        for measure_name, (low, high) in ranges.items():
            person_val = getattr(measurements, measure_name, None)
            if person_val is not None:
                checks += 1
                if low <= person_val <= high:
                    score += 1

        if checks > 0:
            scores[size_key] = (score, checks)
        else:
            scores[size_key] = (0, 0)

        total_checks = max(total_checks, checks)

    if total_checks == 0:
        return None, debug_msgs

    best_size = max(scores, key=lambda s: scores[s][0] / max(scores[s][1], 1) if scores[s][1] > 0 else 0)

    best_score, best_checks = scores[best_size]
    for s, (sc, ch) in scores.items():
        if ch > 0:
            debug_msgs.append(f"{s.value}: {sc}/{ch} matched")

    if best_checks > 0 and best_score / best_checks < 0.3:
        return None, debug_msgs + ["Measurements do not clearly match any standard size"]

    return best_size, debug_msgs


def check_clothing_size(
    clothing: ClothingItemInput,
    person_measurements: Optional[BodyMeasurements],
) -> SizingCheckResult:
    mismatches = []

    if clothing.special_item != SpecialItem.NONE:
        rule = SPECIAL_ITEM_SIZE_RULES.get(clothing.special_item, "")
        return SizingCheckResult(
            passed=True,
            message=f"Special item '{clothing.special_item.value}' — {rule}",
            clothing_size=clothing.size.value if clothing.size else "N/A",
        )

    if clothing.size == StandardSize.ONE_SIZE or clothing.size is None:
        return SizingCheckResult(
            passed=True,
            message="One-size or unspecified size — cannot validate fit",
            clothing_size=clothing.size.value if clothing.size else "unspecified",
        )

    if person_measurements is None:
        return SizingCheckResult(
            passed=False,
            message="Person measurements required for size validation",
            clothing_size=clothing.size.value,
        )

    estimated_size, debug = estimate_person_size(person_measurements)

    relevant_ranges = SIZE_CHART.get(clothing.size, {})

    for measure_name, (low, high) in relevant_ranges.items():
        person_val = getattr(person_measurements, measure_name, None)
        if person_val is not None and not (low <= person_val <= high):
            mismatches.append(
                f"{measure_name.replace('_cm', '')}: person={person_val}cm, "
                f"size {clothing.size.value} expects {low}-{high}cm"
            )

    if estimated_size and estimated_size != clothing.size and mismatches:
        return SizingCheckResult(
            passed=False,
            message=(
                f"Clothing size '{clothing.size.value}' does not match "
                f"estimated person size '{estimated_size.value}'."
            ),
            estimated_person_size=estimated_size.value,
            clothing_size=clothing.size.value,
            mismatches=mismatches,
        )

    if mismatches:
        return SizingCheckResult(
            passed=False,
            message="Clothing size does not fit person's measurements.",
            estimated_person_size=estimated_size.value if estimated_size else None,
            clothing_size=clothing.size.value,
            mismatches=mismatches,
        )

    return SizingCheckResult(
        passed=True,
        message="Size match confirmed.",
        estimated_person_size=estimated_size.value if estimated_size else None,
        clothing_size=clothing.size.value,
    )


def get_special_item_handling(clothing: ClothingItemInput) -> dict:
    if clothing.special_item == SpecialItem.TIE:
        return {
            "requires_neck_measurement": False,
            "sizing_note": "Ties are universal length (~145cm). Focus on color/pattern matching.",
            "worn_with": "collared shirt",
            "override_size_check": True,
        }
    elif clothing.special_item == SpecialItem.BELT:
        return {
            "requires_waist_measurement": True,
            "sizing_note": "Belt size = waist circumference. Most belts are adjustable.",
            "worn_with": "pants or dress",
            "override_size_check": True,
        }
    elif clothing.special_item == SpecialItem.SCARF:
        return {
            "requires_neck_measurement": False,
            "sizing_note": "Scarves are one-size. Style determines fit.",
            "worn_with": "outerwear or shirts",
            "override_size_check": True,
        }
    elif clothing.special_item == SpecialItem.HAT:
        return {
            "requires_head_measurement": True,
            "sizing_note": "Hat size is head circumference in cm.",
            "worn_with": "any outfit",
            "override_size_check": False,
        }
    elif clothing.special_item == SpecialItem.GLOVES:
        return {
            "requires_hand_measurement": True,
            "sizing_note": "Glove size based on hand circumference around knuckles.",
            "worn_with": "outerwear",
            "override_size_check": False,
        }
    elif clothing.special_item == SpecialItem.SUSPENDERS:
        return {
            "requires_waist_measurement": False,
            "sizing_note": "Suspenders are adjustable and fit most body types.",
            "worn_with": "pants",
            "override_size_check": True,
        }
    else:
        return {
            "requires_any_measurement": True,
            "sizing_note": "Standard clothing sizing rules apply.",
            "override_size_check": False,
        }
