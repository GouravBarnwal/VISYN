import json
from pathlib import Path


CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

REVIEW_MULTIPLIER = 1.10

LOCALIZATION_METHOD = (
    "hybrid_alpha_0.75_p95_largest_component"
)


def load_calibration():
    with open(
        CALIBRATION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def get_threshold(calibration, category):
    candidates = calibration["categories"][category][
        "percentile_candidates"
    ]

    for candidate in candidates:
        if candidate["percentile"] == 99.0:
            return candidate["threshold"]

    raise ValueError(
        f"P99 threshold not found for {category}"
    )


def classify_score(score, threshold):
    review_threshold = (
        threshold * REVIEW_MULTIPLIER
    )

    if score < threshold:
        decision = "PASS"
    elif score < review_threshold:
        decision = "REVIEW"
    else:
        decision = "FAIL"

    return decision, review_threshold


def build_inspection_result(
    category,
    anomaly_score,
    evidence=None,
):
    calibration = load_calibration()

    threshold = get_threshold(
        calibration,
        category
    )

    decision, review_threshold = classify_score(
        anomaly_score,
        threshold
    )

    result = {
        "category": category,
        "anomaly_score": float(anomaly_score),
        "threshold": float(threshold),
        "review_threshold": float(review_threshold),
        "decision": decision,
        "score_aggregation": "TOP5",
    }

    # Localization is explanatory evidence.
    # It does not alter the classification decision.
    if decision in ("REVIEW", "FAIL"):

        result["localization"] = {
            "method": LOCALIZATION_METHOD,
            "available": evidence is not None,
            "evidence": evidence,
        }

    else:

        result["localization"] = {
            "method": LOCALIZATION_METHOD,
            "available": False,
            "evidence": None,
        }

    return result


if __name__ == "__main__":

    print("=" * 72)
    print("VISYN — INSPECTION RESULT TEST")
    print("=" * 72)
    print()

    test_cases = [
        ("bottle", 0.080000),
        ("bottle", 0.115000),
        ("bottle", 0.200000),

        ("hazelnut", 0.250000),
        ("hazelnut", 0.300000),
        ("hazelnut", 0.500000),

        ("cable", 0.220000),
        ("cable", 0.260000),
        ("cable", 0.500000),

        ("capsule", 0.100000),
        ("capsule", 0.130000),
        ("capsule", 0.400000),

        ("screw", 0.170000),
        ("screw", 0.220000),
        ("screw", 0.500000),

        ("metal_nut", 0.220000),
        ("metal_nut", 0.280000),
        ("metal_nut", 0.700000),
    ]

    for category, score in test_cases:

        result = build_inspection_result(
            category,
            score,
            evidence={
                "bounding_box": {
                    "x": 100,
                    "y": 100,
                    "width": 200,
                    "height": 200,
                }
            }
            if score >= get_threshold(
                load_calibration(),
                category
            )
            else None
        )

        print(
            f"{category:12s} "
            f"score={score:.6f} "
            f"→ {result['decision']}"
        )

        if result["localization"]["available"]:
            print(
                " " * 4
                + "localization evidence attached"
            )

    print()
    print("=" * 72)
    print("INSPECTION RESULT TEST COMPLETE")
    print("=" * 72)