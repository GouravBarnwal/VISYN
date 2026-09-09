from pathlib import Path
import json

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

NORMAL_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_normal_dev_scores.json"
)

DEFECT_SCORES_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day10_category_metrics.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing artifact: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def calculate_metrics(
    normal_scores,
    defect_scores,
):

    normal_scores = np.asarray(
        normal_scores,
        dtype=np.float64,
    )

    defect_scores = np.asarray(
        defect_scores,
        dtype=np.float64,
    )

    y_true = np.concatenate([
        np.zeros(
            len(normal_scores),
            dtype=np.int32,
        ),
        np.ones(
            len(defect_scores),
            dtype=np.int32,
        ),
    ])

    y_score = np.concatenate([
        normal_scores,
        defect_scores,
    ])

    auroc = roc_auc_score(
        y_true,
        y_score,
    )

    average_precision = (
        average_precision_score(
            y_true,
            y_score,
        )
    )

    return {
        "normal_count": int(
            len(normal_scores)
        ),
        "defect_count": int(
            len(defect_scores)
        ),
        "auroc": float(
            auroc
        ),
        "average_precision": float(
            average_precision
        ),
        "normal_mean": float(
            np.mean(normal_scores)
        ),
        "defect_mean": float(
            np.mean(defect_scores)
        ),
        "normal_median": float(
            np.median(normal_scores)
        ),
        "defect_median": float(
            np.median(defect_scores)
        ),
        "normal_min": float(
            np.min(normal_scores)
        ),
        "normal_max": float(
            np.max(normal_scores)
        ),
        "defect_min": float(
            np.min(defect_scores)
        ),
        "defect_max": float(
            np.max(defect_scores)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — DAY 10 CATEGORY-LEVEL EVALUATION"
    )
    print("=" * 72)

    print()
    print(
        "Model: MobileNetV3-Small L8"
    )

    print(
        "Distance: Euclidean (torch.cdist)"
    )

    print(
        "Aggregation: TOP5"
    )

    print(
        "Evaluation split: development only"
    )

    print(
        "Final locked defect set: NOT USED"
    )

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    normal_data = load_json(
        NORMAL_SCORES_PATH
    )

    defect_data = load_json(
        DEFECT_SCORES_PATH
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if normal_data.get(
        "distance_metric"
    ) != "euclidean_cdist_p2":

        raise RuntimeError(
            "Normal scores are not using "
            "Euclidean cdist."
        )

    if normal_data.get(
        "aggregation"
    ) != "TOP5":

        raise RuntimeError(
            "Normal scores are not using TOP5."
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        "model": (
            "MobileNetV3-Small-L8"
        ),
        "distance_metric": (
            "euclidean_cdist_p2"
        ),
        "aggregation": "TOP5",
        "split": (
            "canonical_development"
        ),
        "final_set_used": False,
        "categories": {},
    }

    category_aurocs = []
    category_aps = []

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("-" * 72)

        # ----------------------------------------------------
        # Normal development scores
        # ----------------------------------------------------

        normal_samples = (
            normal_data[
                "categories"
            ][category][
                "samples"
            ]
        )

        normal_scores = [
            float(
                sample["score"]
            )
            for sample in normal_samples
        ]

        # ----------------------------------------------------
        # Defect development scores
        # ----------------------------------------------------

        category_defect_data = (
            defect_data[
                category
            ]
        )

        if not isinstance(
            category_defect_data,
            dict,
        ):

            raise RuntimeError(
                f"Unexpected defect data "
                f"for {category}: "
                f"{type(category_defect_data)}"
            )

        if "defect_dev" not in category_defect_data:

            raise RuntimeError(
                f"Missing defect_dev data "
                f"for {category}"
            )

        defect_samples = (
            category_defect_data[
                "defect_dev"
            ]
        )

        defect_scores = [
            float(
                sample["top5"]
            )
            for sample in defect_samples
        ]

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            normal_scores,
            defect_scores,
        )

        results[
            "categories"
        ][category] = metrics

        category_aurocs.append(
            metrics["auroc"]
        )

        category_aps.append(
            metrics["average_precision"]
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Normal samples: "
            f"{metrics['normal_count']}"
        )

        print(
            f"Defect samples: "
            f"{metrics['defect_count']}"
        )

        print(
            f"AUROC: "
            f"{metrics['auroc']:.4f}"
        )

        print(
            f"Average Precision: "
            f"{metrics['average_precision']:.4f}"
        )

        print(
            f"Normal median: "
            f"{metrics['normal_median']:.6f}"
        )

        print(
            f"Defect median: "
            f"{metrics['defect_median']:.6f}"
        )

    # ========================================================
    # MACRO RESULTS
    # ========================================================

    macro_auroc = float(
        np.mean(
            category_aurocs
        )
    )

    macro_ap = float(
        np.mean(
            category_aps
        )
    )

    results[
        "macro"
    ] = {
        "auroc": macro_auroc,
        "average_precision": macro_ap,
    }

    # --------------------------------------------------------
    # Pooled evaluation
    # --------------------------------------------------------

    pooled_normal = []
    pooled_defect = []

    for category in CATEGORIES:

        normal_samples = (
            normal_data[
                "categories"
            ][category][
                "samples"
            ]
        )

        pooled_normal.extend([
            float(
                sample["score"]
            )
            for sample in normal_samples
        ])

        defect_samples = (
            defect_data[
                category
            ][
                "defect_dev"
            ]
        )

        pooled_defect.extend([
            float(
                sample["top5"]
            )
            for sample in defect_samples
        ])

    pooled_metrics = calculate_metrics(
        pooled_normal,
        pooled_defect,
    )

    results[
        "pooled"
    ] = pooled_metrics

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
    print(
        "DAY 10 CATEGORY EVALUATION SUMMARY"
    )
    print("=" * 72)

    print()

    print(
        f"{'Category':<15}"
        f"{'AUROC':>10}"
        f"{'AP':>10}"
    )

    print("-" * 40)

    for category in CATEGORIES:

        metrics = results[
            "categories"
        ][category]

        print(
            f"{category:<15}"
            f"{metrics['auroc']:>10.4f}"
            f"{metrics['average_precision']:>10.4f}"
        )

    print("-" * 40)

    print(
        f"{'MACRO':<15}"
        f"{macro_auroc:>10.4f}"
        f"{macro_ap:>10.4f}"
    )

    print()

    print(
        f"Pooled AUROC: "
        f"{pooled_metrics['auroc']:.4f}"
    )

    print(
        f"Pooled AP: "
        f"{pooled_metrics['average_precision']:.4f}"
    )

    print()

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":

    main()