import json
from pathlib import Path
from collections import Counter

CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

NORMAL_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

DEFECT_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/decision_policy_dev_results.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

REVIEW_MULTIPLIER = 1.10


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_p99_threshold(calibration, category):
    candidates = calibration["categories"][category][
        "percentile_candidates"
    ]

    for candidate in candidates:
        if candidate["percentile"] == 99.0:
            return candidate["threshold"]

    raise ValueError(
        f"P99 threshold not found for {category}"
    )


def classify(score, threshold):
    review_threshold = threshold * REVIEW_MULTIPLIER

    if score < threshold:
        return "PASS"

    if score < review_threshold:
        return "REVIEW"

    return "FAIL"


def main():

    calibration = load_json(CALIBRATION_PATH)
    normal = load_json(NORMAL_PATH)
    defects = load_json(DEFECT_PATH)

    results = {}

    total_normal = Counter()
    total_defect = Counter()

    print("=" * 72)
    print("VISIONFORGE — FULL DEVELOPMENT DECISION POLICY EVALUATION")
    print("=" * 72)
    print()
    print("Score: TOP5")
    print("Threshold: category-specific P99")
    print("Review multiplier: 1.10x")
    print("Final defect set: NOT USED")
    print()

    for category in CATEGORIES:

        threshold = get_p99_threshold(
            calibration,
            category
        )

        review_threshold = (
            threshold * REVIEW_MULTIPLIER
        )

        normal_samples = normal[
            "categories"
        ][category]["scores"]

        defect_samples = defects[
            category
        ]["defect_dev"]

        normal_decisions = Counter()
        defect_decisions = Counter()

        for score in normal_samples:

            decision = classify(
                score,
                threshold
            )

            normal_decisions[decision] += 1
            total_normal[decision] += 1

        for sample in defect_samples:

            decision = classify(
                sample["top5"],
                threshold
            )

            defect_decisions[decision] += 1
            total_defect[decision] += 1

        normal_total = len(normal_samples)
        defect_total = len(defect_samples)

        print("-" * 72)
        print(f"CATEGORY: {category}")
        print()
        print(f"P99 threshold:     {threshold:.6f}")
        print(
            f"Review threshold:  "
            f"{review_threshold:.6f}"
        )

        print()
        print("NORMAL DEVELOPMENT")
        print(
            f"  PASS:   "
            f"{normal_decisions['PASS']:3d}/"
            f"{normal_total}"
        )
        print(
            f"  REVIEW: "
            f"{normal_decisions['REVIEW']:3d}/"
            f"{normal_total}"
        )
        print(
            f"  FAIL:   "
            f"{normal_decisions['FAIL']:3d}/"
            f"{normal_total}"
        )

        print()
        print("DEFECT DEVELOPMENT")
        print(
            f"  PASS:   "
            f"{defect_decisions['PASS']:3d}/"
            f"{defect_total}"
        )
        print(
            f"  REVIEW: "
            f"{defect_decisions['REVIEW']:3d}/"
            f"{defect_total}"
        )
        print(
            f"  FAIL:   "
            f"{defect_decisions['FAIL']:3d}/"
            f"{defect_total}"
        )

        results[category] = {
            "threshold": threshold,
            "review_threshold": review_threshold,
            "normal": {
                "total": normal_total,
                "PASS": normal_decisions["PASS"],
                "REVIEW": normal_decisions["REVIEW"],
                "FAIL": normal_decisions["FAIL"],
            },
            "defect": {
                "total": defect_total,
                "PASS": defect_decisions["PASS"],
                "REVIEW": defect_decisions["REVIEW"],
                "FAIL": defect_decisions["FAIL"],
            },
        }

    normal_total_count = sum(total_normal.values())
    defect_total_count = sum(total_defect.values())

    print()
    print("=" * 72)
    print("POOLED DEVELOPMENT RESULTS")
    print("=" * 72)
    print()

    print("NORMAL DEVELOPMENT")
    print(
        f"  PASS:   {total_normal['PASS']:3d}/"
        f"{normal_total_count}"
    )
    print(
        f"  REVIEW: {total_normal['REVIEW']:3d}/"
        f"{normal_total_count}"
    )
    print(
        f"  FAIL:   {total_normal['FAIL']:3d}/"
        f"{normal_total_count}"
    )

    print()
    print("DEFECT DEVELOPMENT")
    print(
        f"  PASS:   {total_defect['PASS']:3d}/"
        f"{defect_total_count}"
    )
    print(
        f"  REVIEW: {total_defect['REVIEW']:3d}/"
        f"{defect_total_count}"
    )
    print(
        f"  FAIL:   {total_defect['FAIL']:3d}/"
        f"{defect_total_count}"
    )

    normal_false_positive_rate = (
        (
            total_normal["REVIEW"]
            + total_normal["FAIL"]
        )
        / normal_total_count
    )

    defect_detection_rate = (
        (
            total_defect["REVIEW"]
            + total_defect["FAIL"]
        )
        / defect_total_count
    )

    defect_fail_rate = (
        total_defect["FAIL"]
        / defect_total_count
    )

    print()
    print(
        f"Normal anomalous-decision rate: "
        f"{normal_false_positive_rate:.4f}"
    )

    print(
        f"Defect detected rate: "
        f"{defect_detection_rate:.4f}"
    )

    print(
        f"Defect direct-FAIL rate: "
        f"{defect_fail_rate:.4f}"
    )

    output = {
        "score": "TOP5",
        "threshold_policy": "P99",
        "review_multiplier": REVIEW_MULTIPLIER,
        "final_defect_set_used": False,
        "categories": results,
        "pooled": {
            "normal": {
                "total": normal_total_count,
                "PASS": total_normal["PASS"],
                "REVIEW": total_normal["REVIEW"],
                "FAIL": total_normal["FAIL"],
            },
            "defect": {
                "total": defect_total_count,
                "PASS": total_defect["PASS"],
                "REVIEW": total_defect["REVIEW"],
                "FAIL": total_defect["FAIL"],
            },
            "normal_anomalous_decision_rate": (
                normal_false_positive_rate
            ),
            "defect_detection_rate": (
                defect_detection_rate
            ),
            "defect_direct_fail_rate": (
                defect_fail_rate
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print("FULL DEVELOPMENT POLICY EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()