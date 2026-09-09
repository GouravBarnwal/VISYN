import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


DINO_PATH = Path(
    "artifacts/evaluation/dino_dev_scores_normalized.json"
)

MOBILE_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/fusion_dev_results.json"
)

METHODS = ["max", "top5", "top10"]

WEIGHTS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


def normalize_path(path):
    return Path(
        path.replace("\\", "/")
    ).as_posix()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_score(item, method):

    if method == "max":
        return float(item["max"])

    if method == "top5":
        return float(item["top5"])

    if method == "top10":
        return float(item["top10"])

    raise ValueError(
        f"Unknown method: {method}"
    )


def build_lookup(items):

    return {
        normalize_path(item["path"]): item
        for item in items
    }


def collect_category_scores(
    dino,
    mobile,
    category,
    method,
):

    dino_normal = build_lookup(
        dino[category]["normal_dev"]
    )

    dino_defect = build_lookup(
        dino[category]["defect_dev"]
    )

    mobile_normal = build_lookup(
        mobile[category]["normal_dev"]
    )

    mobile_defect = build_lookup(
        mobile[category]["defect_dev"]
    )

    # -------------------------------------------------------------
    # Verify that both models use EXACTLY the same images.
    # -------------------------------------------------------------

    if set(dino_normal) != set(mobile_normal):

        only_dino = sorted(
            set(dino_normal) - set(mobile_normal)
        )

        only_mobile = sorted(
            set(mobile_normal) - set(dino_normal)
        )

        raise RuntimeError(
            f"Normal image mismatch: {category}\n"
            f"Only DINO: {only_dino[:5]}\n"
            f"Only MobileNet: {only_mobile[:5]}"
        )

    if set(dino_defect) != set(mobile_defect):

        only_dino = sorted(
            set(dino_defect) - set(mobile_defect)
        )

        only_mobile = sorted(
            set(mobile_defect) - set(dino_defect)
        )

        raise RuntimeError(
            f"Defect image mismatch: {category}\n"
            f"Only DINO: {only_dino[:5]}\n"
            f"Only MobileNet: {only_mobile[:5]}"
        )

    normal_paths = sorted(dino_normal)
    defect_paths = sorted(dino_defect)

    dino_normal_scores = np.array(
        [
            get_score(
                dino_normal[path],
                method,
            )
            for path in normal_paths
        ],
        dtype=np.float64,
    )

    dino_defect_scores = np.array(
        [
            get_score(
                dino_defect[path],
                method,
            )
            for path in defect_paths
        ],
        dtype=np.float64,
    )

    mobile_normal_scores = np.array(
        [
            get_score(
                mobile_normal[path],
                method,
            )
            for path in normal_paths
        ],
        dtype=np.float64,
    )

    mobile_defect_scores = np.array(
        [
            get_score(
                mobile_defect[path],
                method,
            )
            for path in defect_paths
        ],
        dtype=np.float64,
    )

    return (
        dino_normal_scores,
        dino_defect_scores,
        mobile_normal_scores,
        mobile_defect_scores,
    )


def robust_scale(normal_scores):

    median = np.median(
        normal_scores
    )

    mad = np.median(
        np.abs(
            normal_scores - median
        )
    )

    scale = 1.4826 * mad

    if scale < 1e-12:
        scale = np.std(
            normal_scores
        )

    if scale < 1e-12:
        scale = 1.0

    return median, scale


def transform(
    scores,
    median,
    scale,
):

    return (
        scores - median
    ) / scale


def main():

    print("=" * 72)
    print(
        "VISIONFORGE — GLOBAL DEVELOPMENT FUSION SELECTION"
    )
    print("=" * 72)

    dino = load_json(
        DINO_PATH
    )

    mobile = load_json(
        MOBILE_PATH
    )

    categories = list(
        dino.keys()
    )

    results = {}

    for method in METHODS:

        print()
        print("=" * 72)
        print(
            f"FUSION METHOD — {method.upper()}"
        )
        print("=" * 72)

        category_data = {}

        # ---------------------------------------------------------
        # Prepare every category.
        # ---------------------------------------------------------

        for category in categories:

            (
                dino_normal,
                dino_defect,
                mobile_normal,
                mobile_defect,
            ) = collect_category_scores(
                dino,
                mobile,
                category,
                method,
            )

            # -----------------------------------------------------
            # Normalization statistics come ONLY from normal_dev.
            # -----------------------------------------------------

            dino_median, dino_scale = (
                robust_scale(
                    dino_normal
                )
            )

            mobile_median, mobile_scale = (
                robust_scale(
                    mobile_normal
                )
            )

            dino_normal_z = transform(
                dino_normal,
                dino_median,
                dino_scale,
            )

            dino_defect_z = transform(
                dino_defect,
                dino_median,
                dino_scale,
            )

            mobile_normal_z = transform(
                mobile_normal,
                mobile_median,
                mobile_scale,
            )

            mobile_defect_z = transform(
                mobile_defect,
                mobile_median,
                mobile_scale,
            )

            y_true = np.concatenate(
                [
                    np.zeros(
                        len(dino_normal)
                    ),
                    np.ones(
                        len(dino_defect)
                    ),
                ]
            )

            mobile_scores = np.concatenate(
                [
                    mobile_normal_z,
                    mobile_defect_z,
                ]
            )

            dino_scores = np.concatenate(
                [
                    dino_normal_z,
                    dino_defect_z,
                ]
            )

            mobile_auc = roc_auc_score(
                y_true,
                mobile_scores,
            )

            dino_auc = roc_auc_score(
                y_true,
                dino_scores,
            )

            category_data[category] = {
                "mobile_auc": float(
                    mobile_auc
                ),
                "dino_auc": float(
                    dino_auc
                ),
                "mobile_normal_z": mobile_normal_z,
                "mobile_defect_z": mobile_defect_z,
                "dino_normal_z": dino_normal_z,
                "dino_defect_z": dino_defect_z,
            }

        # ---------------------------------------------------------
        # Search ONE global fusion weight.
        # ---------------------------------------------------------

        global_results = []

        for mobile_weight in WEIGHTS:

            dino_weight = (
                1.0 - mobile_weight
            )

            category_aucs = {}

            for category in categories:

                data = category_data[
                    category
                ]

                fused_normal = (
                    mobile_weight
                    * data["mobile_normal_z"]
                    +
                    dino_weight
                    * data["dino_normal_z"]
                )

                fused_defect = (
                    mobile_weight
                    * data["mobile_defect_z"]
                    +
                    dino_weight
                    * data["dino_defect_z"]
                )

                y_true = np.concatenate(
                    [
                        np.zeros(
                            len(fused_normal)
                        ),
                        np.ones(
                            len(fused_defect)
                        ),
                    ]
                )

                fused_scores = np.concatenate(
                    [
                        fused_normal,
                        fused_defect,
                    ]
                )

                auc = roc_auc_score(
                    y_true,
                    fused_scores,
                )

                category_aucs[
                    category
                ] = float(auc)

            macro_auc = float(
                np.mean(
                    list(
                        category_aucs.values()
                    )
                )
            )

            global_results.append(
                {
                    "mobile_weight": mobile_weight,
                    "dino_weight": dino_weight,
                    "macro_auroc": macro_auc,
                    "category_auroc": category_aucs,
                }
            )

        best = max(
            global_results,
            key=lambda x: x["macro_auroc"]
        )

        mobile_macro = float(
            np.mean(
                [
                    category_data[c][
                        "mobile_auc"
                    ]
                    for c in categories
                ]
            )
        )

        dino_macro = float(
            np.mean(
                [
                    category_data[c][
                        "dino_auc"
                    ]
                    for c in categories
                ]
            )
        )

        print()
        print(
            "GLOBAL WEIGHT SEARCH"
        )
        print("-" * 72)

        for result in global_results:

            print(
                f"M={result['mobile_weight']:.1f} "
                f"D={result['dino_weight']:.1f} "
                f"| Macro AUROC = "
                f"{result['macro_auroc']:.4f}"
            )

        print("-" * 72)

        print(
            f"MobileNet only : "
            f"{mobile_macro:.4f}"
        )

        print(
            f"DINOv2 only    : "
            f"{dino_macro:.4f}"
        )

        print(
            f"BEST GLOBAL    : "
            f"M={best['mobile_weight']:.1f} "
            f"D={best['dino_weight']:.1f} "
            f"| {best['macro_auroc']:.4f}"
        )

        results[method] = {
            "mobile_macro_auroc": mobile_macro,
            "dino_macro_auroc": dino_macro,
            "weight_search": global_results,
            "best_global": best,
        }

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

    print()
    print("=" * 72)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()