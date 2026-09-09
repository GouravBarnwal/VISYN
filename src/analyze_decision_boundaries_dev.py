import json
from pathlib import Path
import numpy as np

NORMAL_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

DEFECT_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

CALIBRATION_PATH = Path(
    "artifacts/evaluation/calibration_candidates.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/decision_boundaries_dev.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

TARGET_PERCENTILE = 99.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_p99_threshold(calibration, category):
    candidates = calibration["categories"][category][
        "percentile_candidates"
    ]

    for candidate in candidates:
        if candidate["percentile"] == TARGET_PERCENTILE:
            return candidate["threshold"]

    raise ValueError(
        f"P{TARGET_PERCENTILE} threshold not found for {category}"
    )


def percentile(values, p):
    return float(np.percentile(values, p))


def main():

    normal = load_json(NORMAL_PATH)
    defects = load_json(DEFECT_PATH)
    calibration = load_json(CALIBRATION_PATH)

    results = {}

    print("=" * 72)
    print("VISIONFORGE — DECISION BOUNDARY ANALYSIS")
    print("=" * 72)
    print()
    print("Normal source: normal development only")
    print("Defect source: defect development only")
    print("Final defect set: NOT USED")
    print("Score aggregation: TOP5")
    print("Calibration: category-specific P99")
    print()

    for category in CATEGORIES:

        normal_scores = np.array(
            normal["categories"][category]["scores"],
            dtype=float
        )

        defect_scores = np.array(
            [
                sample["top5"]
                for sample in defects[category]["defect_dev"]
            ],
            dtype=float
        )

        threshold = get_p99_threshold(
            calibration,
            category
        )

        normal_margins = normal_scores - threshold
        defect_margins = defect_scores - threshold

        closest_normal_margin = float(
            np.max(normal_margins)
        )

        closest_defect_margin = float(
            np.min(defect_margins)
        )

        multipliers = [1.00, 1.10, 1.25, 1.50, 2.00]

        multiplier_results = {}

        print("-" * 72)
        print(f"CATEGORY: {category}")
        print()

        print(f"P99 threshold:       {threshold:.6f}")
        print(
            f"Normal mean:         "
            f"{np.mean(normal_scores):.6f}"
        )
        print(
            f"Normal P95:          "
            f"{percentile(normal_scores, 95):.6f}"
        )
        print(
            f"Normal P99:          "
            f"{percentile(normal_scores, 99):.6f}"
        )
        print(
            f"Normal maximum:      "
            f"{np.max(normal_scores):.6f}"
        )

        print()

        print(
            f"Defect minimum:      "
            f"{np.min(defect_scores):.6f}"
        )
        print(
            f"Defect median:       "
            f"{np.median(defect_scores):.6f}"
        )
        print(
            f"Defect P25:          "
            f"{percentile(defect_scores, 25):.6f}"
        )
        print(
            f"Defect maximum:      "
            f"{np.max(defect_scores):.6f}"
        )

        print()

        print(
            f"Closest normal margin: "
            f"{closest_normal_margin:+.6f}"
        )

        print(
            f"Closest defect margin: "
            f"{closest_defect_margin:+.6f}"
        )

        print()
        print("CANDIDATE REVIEW BANDS")
        print()

        for multiplier in multipliers:

            review_upper = threshold * multiplier

            normal_pass = np.sum(
                normal_scores < threshold
            )

            normal_review = np.sum(
                (normal_scores >= threshold)
                & (normal_scores < review_upper)
            )

            normal_fail = np.sum(
                normal_scores >= review_upper
            )

            defect_review = np.sum(
                (defect_scores >= threshold)
                & (defect_scores < review_upper)
            )

            defect_fail = np.sum(
                defect_scores >= review_upper
            )

            defect_fail_recall = (
                defect_fail / len(defect_scores)
            )

            print(
                f"{multiplier:>4.2f}x  "
                f"upper={review_upper:.6f}  "
                f"normal_pass={normal_pass:3d}  "
                f"normal_review={normal_review:3d}  "
                f"normal_fail={normal_fail:3d}  "
                f"defect_review={defect_review:3d}  "
                f"defect_fail={defect_fail:3d}  "
                f"defect_fail_recall={defect_fail_recall:.4f}"
            )

            multiplier_results[
                f"{multiplier:.2f}x"
            ] = {
                "review_upper": float(review_upper),
                "normal_pass": int(normal_pass),
                "normal_review": int(normal_review),
                "normal_fail": int(normal_fail),
                "defect_review": int(defect_review),
                "defect_fail": int(defect_fail),
                "defect_fail_recall": float(
                    defect_fail_recall
                ),
            }

        results[category] = {
            "threshold": float(threshold),

            "normal": {
                "count": int(len(normal_scores)),
                "mean": float(np.mean(normal_scores)),
                "p95": percentile(normal_scores, 95),
                "p99": percentile(normal_scores, 99),
                "max": float(np.max(normal_scores)),
            },

            "defect": {
                "count": int(len(defect_scores)),
                "min": float(np.min(defect_scores)),
                "p25": percentile(defect_scores, 25),
                "median": float(np.median(defect_scores)),
                "max": float(np.max(defect_scores)),
            },

            "closest_normal_margin": closest_normal_margin,
            "closest_defect_margin": closest_defect_margin,

            "candidate_review_bands": multiplier_results,
        }

        print()

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
            {
                "score": "TOP5",
                "calibration": "P99",
                "final_defect_set_used": False,
                "categories": results,
            },
            f,
            indent=2
        )

    print("=" * 72)
    print("DECISION BOUNDARY ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()