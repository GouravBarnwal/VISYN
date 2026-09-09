from pathlib import Path
import json

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

NORMAL_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

CALIBRATION_PATH = Path(
    "artifacts/evaluation/production_thresholds.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day10_defect_type_metrics.json"
)

REVIEW_MULTIPLIER = 1.10

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# LOADERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# METRICS
# ============================================================

def calculate_auc_metrics(normal_scores, defect_scores):
    normal_scores = np.asarray(
        normal_scores,
        dtype=np.float64,
    )

    defect_scores = np.asarray(
        defect_scores,
        dtype=np.float64,
    )

    y_true = np.concatenate(
        [
            np.zeros(len(normal_scores), dtype=np.int32),
            np.ones(len(defect_scores), dtype=np.int32),
        ]
    )

    y_score = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    return {
        "auroc": float(
            roc_auc_score(y_true, y_score)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),
    }


# ============================================================
# DECISION POLICY
# ============================================================

def classify_score(score, threshold):
    review_threshold = (
        threshold * REVIEW_MULTIPLIER
    )

    if score < threshold:
        return "PASS"

    if score < review_threshold:
        return "REVIEW"

    return "FAIL"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("VISYN — DAY 10 DEFECT-TYPE ANALYSIS")
    print("=" * 72)

    print()
    print("Model: MobileNetV3-Small L8")
    print("Distance: Euclidean (torch.cdist)")
    print("Aggregation: TOP5")
    print("Evaluation split: development only")
    print("Final locked defect set: NOT USED")
    print()

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    normal_data = load_json(
        NORMAL_SCORES_PATH
    )

    defect_data = load_json(
        DEFECT_SCORES_PATH
    )

    calibration_data = load_json(
        CALIBRATION_PATH
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if normal_data.get(
        "distance_metric"
    ) != "euclidean_cdist_p2":
        raise RuntimeError(
            "Normal score artifact is not using "
            "Euclidean cdist."
        )

    if normal_data.get(
        "aggregation"
    ) != "TOP5":
        raise RuntimeError(
            "Normal score artifact is not using TOP5."
        )

    if calibration_data.get(
        "distance_metric"
    ) != "euclidean_cdist_p2":
        raise RuntimeError(
            "Calibration artifact is not using "
            "Euclidean cdist."
        )

    if calibration_data.get(
        "aggregation"
    ) != "TOP5":
        raise RuntimeError(
            "Calibration artifact is not using TOP5."
        )

    if calibration_data.get(
        "selected_percentile"
    ) != 99.0:
        raise RuntimeError(
            "Production calibration is not using P99."
        )

    # --------------------------------------------------------
    # Results structure
    # --------------------------------------------------------

    results = {
        "model": "MobileNetV3-Small-L8",
        "distance_metric": "euclidean_cdist_p2",
        "aggregation": "TOP5",
        "split": "canonical_development",
        "final_set_used": False,
        "calibration": "category_specific_P99",
        "review_multiplier": REVIEW_MULTIPLIER,
        "categories": {},
    }

    all_type_aurocs = []
    all_type_aps = []

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:
        print()
        print("#" * 72)
        print(f"CATEGORY: {category}")
        print("#" * 72)

        # ----------------------------------------------------
        # Normal development scores
        # ----------------------------------------------------

        normal_samples = (
            normal_data["categories"][category]["samples"]
        )

        normal_scores = [
            float(sample["score"])
            for sample in normal_samples
        ]

        # ----------------------------------------------------
        # Production P99 threshold
        # ----------------------------------------------------

        threshold = float(
            calibration_data["categories"][category]["threshold"]
        )

        review_threshold = (
            threshold * REVIEW_MULTIPLIER
        )

        # ----------------------------------------------------
        # Defect development samples
        # ----------------------------------------------------

        defect_samples = (
            defect_data[category]["defect_dev"]
        )

        defect_types = sorted(
            {
                sample["defect_type"]
                for sample in defect_samples
            }
        )

        results["categories"][category] = {
            "p99_threshold": threshold,
            "review_threshold": review_threshold,
            "defect_types": {},
        }

        print()
        print(f"P99 threshold: {threshold:.6f}")
        print(
            f"Review threshold: "
            f"{review_threshold:.6f}"
        )

        # ====================================================
        # DEFECT TYPE LOOP
        # ====================================================

        for defect_type in defect_types:
            samples = [
                sample
                for sample in defect_samples
                if sample["defect_type"] == defect_type
            ]

            scores = [
                float(sample["top5"])
                for sample in samples
            ]

            # ------------------------------------------------
            # Score-quality metrics
            # ------------------------------------------------

            auc_metrics = calculate_auc_metrics(
                normal_scores,
                scores,
            )

            # ------------------------------------------------
            # Threshold detection
            # ------------------------------------------------

            decisions = [
                classify_score(
                    score,
                    threshold,
                )
                for score in scores
            ]

            pass_count = decisions.count("PASS")
            review_count = decisions.count("REVIEW")
            fail_count = decisions.count("FAIL")

            detected_count = (
                review_count + fail_count
            )

            detection_rate = (
                detected_count / len(scores)
            )

            fail_rate = (
                fail_count / len(scores)
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            type_result = {
                "sample_count": len(scores),
                "auroc": auc_metrics["auroc"],
                "average_precision": (
                    auc_metrics["average_precision"]
                ),
                "score_mean": float(
                    np.mean(scores)
                ),
                "score_median": float(
                    np.median(scores)
                ),
                "score_min": float(
                    np.min(scores)
                ),
                "score_max": float(
                    np.max(scores)
                ),
                "detected_count": detected_count,
                "detection_rate": float(
                    detection_rate
                ),
                "direct_fail_count": fail_count,
                "direct_fail_rate": float(
                    fail_rate
                ),
                "pass_count": pass_count,
                "review_count": review_count,
                "decision_distribution": {
                    "PASS": pass_count,
                    "REVIEW": review_count,
                    "FAIL": fail_count,
                },
            }

            results["categories"][category][
                "defect_types"
            ][defect_type] = type_result

            all_type_aurocs.append(
                auc_metrics["auroc"]
            )

            all_type_aps.append(
                auc_metrics["average_precision"]
            )

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print(
                f"{defect_type:<24}"
                f"AUROC {auc_metrics['auroc']:.4f}   "
                f"AP {auc_metrics['average_precision']:.4f}   "
                f"Detect {detection_rate:.3f}   "
                f"FAIL {fail_rate:.3f}"
            )

    # ========================================================
    # MACRO DEFECT-TYPE RESULTS
    # ========================================================

    results["macro_defect_type"] = {
        "defect_type_count": len(all_type_aurocs),
        "auroc": float(
            np.mean(all_type_aurocs)
        ),
        "average_precision": float(
            np.mean(all_type_aps)
        ),
    }

    # ========================================================
    # IDENTIFY WEAKEST TYPES
    # ========================================================

    flattened = []

    for category in CATEGORIES:
        defect_types = (
            results["categories"][category]["defect_types"]
        )

        for defect_type, metrics in defect_types.items():
            flattened.append(
                {
                    "category": category,
                    "defect_type": defect_type,
                    "auroc": metrics["auroc"],
                    "average_precision": (
                        metrics["average_precision"]
                    ),
                    "detection_rate": (
                        metrics["detection_rate"]
                    ),
                    "direct_fail_rate": (
                        metrics["direct_fail_rate"]
                    ),
                }
            )

    weakest_by_auroc = sorted(
        flattened,
        key=lambda x: x["auroc"],
    )

    weakest_by_detection = sorted(
        flattened,
        key=lambda x: x["detection_rate"],
    )

    results["weakest_by_auroc"] = (
        weakest_by_auroc[:10]
    )

    results["weakest_by_detection"] = (
        weakest_by_detection[:10]
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("DEFECT-TYPE SUMMARY")
    print("=" * 72)

    print()
    print(
        f"Defect types evaluated: "
        f"{len(all_type_aurocs)}"
    )

    print(
        f"Macro defect-type AUROC: "
        f"{results['macro_defect_type']['auroc']:.4f}"
    )

    print(
        f"Macro defect-type AP: "
        f"{results['macro_defect_type']['average_precision']:.4f}"
    )

    print()
    print("10 weakest defect types by AUROC:")

    for item in weakest_by_auroc[:10]:
        print(
            f"{item['category']:<12}"
            f"{item['defect_type']:<24}"
            f"AUROC {item['auroc']:.4f}"
        )

    print()
    print("10 weakest defect types by detection rate:")

    for item in weakest_by_detection[:10]:
        print(
            f"{item['category']:<12}"
            f"{item['defect_type']:<24}"
            f"Detect {item['detection_rate']:.3f}"
        )

    print()
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()