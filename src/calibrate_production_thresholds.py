from pathlib import Path
import json

import numpy as np


# ============================================================
# CONFIG
# ============================================================

NORMAL_DEV_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/production_thresholds.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

SELECTED_PERCENTILE = 99.0


# ============================================================
# HELPERS
# ============================================================

def percentile_threshold(
    scores,
    percentile,
):
    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot calculate threshold from empty scores."
        )

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — PRODUCTION SCORE CALIBRATION"
    )
    print(
        "NORMAL DEVELOPMENT ONLY"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load corrected normal-development scores
    # --------------------------------------------------------

    if not NORMAL_DEV_SCORES_PATH.exists():

        raise FileNotFoundError(
            f"Missing normal-development score artifact:\n"
            f"{NORMAL_DEV_SCORES_PATH}\n\n"
            "Run:\n"
            "python src\\mobilenet_l8_normal_dev_scoring.py"
        )

    with open(
        NORMAL_DEV_SCORES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        score_data = json.load(f)

    print(
        f"Loaded scores from: "
        f"{NORMAL_DEV_SCORES_PATH}"
    )

    print(
        f"Distance metric: "
        f"{score_data.get('distance_metric')}"
    )

    print(
        f"Aggregation: "
        f"{score_data.get('aggregation')}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if score_data.get(
        "distance_metric"
    ) != "euclidean_cdist_p2":

        raise RuntimeError(
            "INVALID SCORE ARTIFACT.\n"
            "Expected distance_metric='euclidean_cdist_p2'.\n"
            "Production calibration must use the same "
            "Euclidean metric as production scoring."
        )

    if score_data.get(
        "aggregation"
    ) != "TOP5":

        raise RuntimeError(
            "INVALID SCORE ARTIFACT.\n"
            "Expected aggregation='TOP5'."
        )

    if "categories" not in score_data:

        raise RuntimeError(
            "Normal-development categories were not found "
            "in the score artifact."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "method": "MobileNetV3-Small-L8",
        "distance_metric": "euclidean_cdist_p2",
        "aggregation": "TOP5",
        "calibration_split": (
            "canonical_normal_development"
        ),
        "seed": score_data.get("seed"),
        "reference_ratio": score_data.get(
            "reference_ratio"
        ),
        "selected_percentile": SELECTED_PERCENTILE,
        "categories": {},
    }

    # ========================================================
    # CATEGORY CALIBRATION
    # ========================================================

    all_normal_scores = []

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("-" * 72)

        category_data = (
            score_data[
                "categories"
            ].get(category)
        )

        if category_data is None:

            raise RuntimeError(
                f"Missing category in normal-development "
                f"scores: {category}"
            )

        scores = category_data.get(
            "scores"
        )

        if not scores:

            raise RuntimeError(
                f"No normal-development scores found "
                f"for {category}."
            )

        values = np.asarray(
            scores,
            dtype=np.float64,
        )

        threshold = percentile_threshold(
            values,
            SELECTED_PERCENTILE,
        )

        flagged = int(
            np.sum(
                values >= threshold
            )
        )

        rate = (
            flagged / len(values)
        )

        all_normal_scores.extend(
            values.tolist()
        )

        output[
            "categories"
        ][category] = {

            "threshold": threshold,

            "percentile": (
                SELECTED_PERCENTILE
            ),

            "normal_development_count": int(
                len(values)
            ),

            "normal_development_min": float(
                np.min(values)
            ),

            "normal_development_max": float(
                np.max(values)
            ),

            "normal_development_mean": float(
                np.mean(values)
            ),

            "normal_development_std": float(
                np.std(values)
            ),

            "normal_development_median": float(
                np.median(values)
            ),

            "flagged_at_threshold": flagged,

            "false_positive_rate": float(
                rate
            ),
        }

        print(
            f"Normal development samples: "
            f"{len(values)}"
        )

        print(
            f"P99 threshold: "
            f"{threshold:.6f}"
        )

        print(
            f"Flagged: "
            f"{flagged}/{len(values)}"
        )

        print(
            f"Rate: "
            f"{rate:.4f}"
        )

    # ========================================================
    # POOLED NORMAL DEVELOPMENT STATISTICS
    # ========================================================

    pooled = np.asarray(
        all_normal_scores,
        dtype=np.float64,
    )

    pooled_threshold = percentile_threshold(
        pooled,
        SELECTED_PERCENTILE,
    )

    pooled_flagged = int(
        np.sum(
            pooled >= pooled_threshold
        )
    )

    pooled_rate = (
        pooled_flagged / len(pooled)
    )

    output[
        "pooled_normal_development"
    ] = {

        "count": int(
            len(pooled)
        ),

        "threshold": pooled_threshold,

        "flagged": pooled_flagged,

        "false_positive_rate": float(
            pooled_rate
        ),

        "min": float(
            np.min(pooled)
        ),

        "max": float(
            np.max(pooled)
        ),

        "mean": float(
            np.mean(pooled)
        ),

        "std": float(
            np.std(pooled)
        ),
    }

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
            output,
            f,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PRODUCTION CALIBRATION COMPLETE"
    )
    print("=" * 72)

    print()
    print(
        "SELECTED PRODUCTION THRESHOLDS"
    )
    print("-" * 72)

    for category in CATEGORIES:

        threshold = (
            output[
                "categories"
            ][category][
                "threshold"
            ]
        )

        print(
            f"{category:<12} "
            f"{threshold:.6f}"
        )

    print()
    print(
        f"Pooled normal-development samples: "
        f"{len(pooled)}"
    )

    print(
        f"Pooled P99 threshold: "
        f"{pooled_threshold:.6f}"
    )

    print(
        f"Pooled flagged: "
        f"{pooled_flagged}/{len(pooled)}"
    )

    print(
        f"Pooled false-positive rate: "
        f"{pooled_rate:.4f}"
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()
    print(
        "Calibration source: "
        "NORMAL DEVELOPMENT ONLY"
    )

    print(
        "Metric: Euclidean torch.cdist(p=2)"
    )

    print(
        "Aggregation: TOP5"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()