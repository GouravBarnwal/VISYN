from pathlib import Path
import json


CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

REVIEW_MULTIPLIER = 1.10


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
        f"P99 threshold not found for category: {category}"
    )


def classify_score(score, threshold):
    review_threshold = threshold * REVIEW_MULTIPLIER

    if score < threshold:
        decision = "PASS"

    elif score < review_threshold:
        decision = "REVIEW"

    else:
        decision = "FAIL"

    return {
        "decision": decision,
        "score": float(score),
        "threshold": float(threshold),
        "review_threshold": float(review_threshold),
    }


def classify(category, score):
    calibration = load_calibration()

    threshold = get_threshold(
        calibration,
        category
    )

    return classify_score(
        score,
        threshold
    )


if __name__ == "__main__":

    print("=" * 72)
    print("VISIONFORGE — DECISION POLICY TEST")
    print("=" * 72)
    print()

    calibration = load_calibration()

    test_scores = {
        "bottle": [0.08, 0.115, 0.20],
        "hazelnut": [0.25, 0.30, 0.50],
        "cable": [0.22, 0.26, 0.50],
        "capsule": [0.10, 0.13, 0.40],
        "screw": [0.17, 0.22, 0.50],
        "metal_nut": [0.22, 0.28, 0.70],
    }

    for category, scores in test_scores.items():

        threshold = get_threshold(
            calibration,
            category
        )

        print(f"CATEGORY: {category}")
        print(
            f"P99 threshold: "
            f"{threshold:.6f}"
        )

        print(
            f"Review threshold: "
            f"{threshold * REVIEW_MULTIPLIER:.6f}"
        )

        for score in scores:

            result = classify_score(
                score,
                threshold
            )

            print(
                f"  score={score:.6f} "
                f"→ {result['decision']}"
            )

        print()

    print("=" * 72)
    print("DECISION POLICY TEST COMPLETE")
    print("=" * 72)