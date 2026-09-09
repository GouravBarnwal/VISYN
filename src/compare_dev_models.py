import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

DINO_PATH = Path(
    "artifacts/evaluation/dino_dev_scores_normalized.json"
)

MOBILE_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)


CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

AGGREGATIONS = [
    "max",
    "top5",
    "top10",
]


# ============================================================
# LOAD
# ============================================================

def load_scores(path):

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file:\n{path}\n"
        )

    with open(path, "r") as f:
        return json.load(f)


dino = load_scores(DINO_PATH)
mobile = load_scores(MOBILE_PATH)


# ============================================================
# PATH NORMALIZATION
# ============================================================

def normalize_path(path):

    # DINO was generated on Linux/Colab:
    #     bottle/train/good/001.png
    #
    # MobileNet was generated on Windows:
    #     bottle\train\good\001.png
    #
    # Convert both to the same canonical representation.

    return Path(
        path.replace("\\", "/")
    ).as_posix()


def get_paths(category, split, data):

    return {
        normalize_path(item["path"])
        for item in data[category][split]
    }


# ============================================================
# VALIDATE SAME IMAGE SET
# ============================================================

print("=" * 72)
print(
    "VISYN — DEVELOPMENT MODEL COMPARISON"
)
print("=" * 72)

print(
    "\nChecking that DINOv2 and MobileNet "
    "use identical images...\n"
)


for category in CATEGORIES:

    for split in [
        "normal_dev",
        "defect_dev",
    ]:

        dino_paths = get_paths(
            category,
            split,
            dino
        )

        mobile_paths = get_paths(
            category,
            split,
            mobile
        )

        if dino_paths != mobile_paths:

            only_dino = sorted(
                dino_paths - mobile_paths
            )

            only_mobile = sorted(
                mobile_paths - dino_paths
            )

            raise RuntimeError(
                f"\nIMAGE SET MISMATCH\n"
                f"Category: {category}\n"
                f"Split: {split}\n"
                f"Only DINO: {only_dino[:5]}\n"
                f"Only MobileNet: {only_mobile[:5]}"
            )

        print(
            f"{category:10s} | "
            f"{split:12s} | "
            f"{len(dino_paths):3d} images | MATCH"
        )


print(
    "\nAll image sets match exactly."
)


# ============================================================
# AUROC
# ============================================================

results = {
    "dino": {},
    "mobilenet": {},
}


def calculate_auroc(
    data,
    category,
    aggregation
):

    normal = data[
        category
    ]["normal_dev"]

    defects = data[
        category
    ]["defect_dev"]

    y_true = (
        [0] * len(normal)
        +
        [1] * len(defects)
    )

    scores = (
        [
            item[aggregation]
            for item in normal
        ]
        +
        [
            item[aggregation]
            for item in defects
        ]
    )

    return roc_auc_score(
        y_true,
        scores
    )


# ============================================================
# CALCULATE
# ============================================================

for model_name, data in [
    ("dino", dino),
    ("mobilenet", mobile),
]:

    for category in CATEGORIES:

        results[
            model_name
        ][category] = {}

        for aggregation in AGGREGATIONS:

            auc = calculate_auroc(
                data,
                category,
                aggregation
            )

            results[
                model_name
            ][category][aggregation] = auc


# ============================================================
# PRINT COMPARISON TABLES
# ============================================================

for aggregation in AGGREGATIONS:

    print()
    print("=" * 72)

    print(
        f"DEVELOPMENT AUROC — "
        f"{aggregation.upper()}"
    )

    print("=" * 72)

    print(
        f"{'Category':12s}"
        f"{'MobileNet L8':>16s}"
        f"{'DINOv2':>16s}"
        f"{'Winner':>16s}"
    )

    print("-" * 72)

    mobile_values = []
    dino_values = []

    for category in CATEGORIES:

        mobile_auc = results[
            "mobilenet"
        ][category][aggregation]

        dino_auc = results[
            "dino"
        ][category][aggregation]

        mobile_values.append(
            mobile_auc
        )

        dino_values.append(
            dino_auc
        )

        if dino_auc > mobile_auc:

            winner = "DINOv2"

        elif mobile_auc > dino_auc:

            winner = "MobileNet"

        else:

            winner = "Tie"

        print(
            f"{category:12s}"
            f"{mobile_auc:16.4f}"
            f"{dino_auc:16.4f}"
            f"{winner:>16s}"
        )


    mobile_mean = np.mean(
        mobile_values
    )

    dino_mean = np.mean(
        dino_values
    )

    print("-" * 72)

    print(
        f"{'Macro mean':12s}"
        f"{mobile_mean:16.4f}"
        f"{dino_mean:16.4f}"
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print()
print("=" * 72)
print(
    "OVERALL DEVELOPMENT SUMMARY"
)
print("=" * 72)


for model_name, display_name in [
    ("mobilenet", "MobileNet L8"),
    ("dino", "DINOv2"),
]:

    print(
        f"\n{display_name}"
    )

    for aggregation in AGGREGATIONS:

        values = [
            results[
                model_name
            ][category][aggregation]
            for category in CATEGORIES
        ]

        print(
            f"  {aggregation.upper():6s}: "
            f"{np.mean(values):.4f}"
        )


# ============================================================
# BEST DEVELOPMENT CONFIGURATION
# ============================================================

best_model = None
best_aggregation = None
best_score = -np.inf


for model_name in results:

    for aggregation in AGGREGATIONS:

        values = [
            results[
                model_name
            ][category][aggregation]
            for category in CATEGORIES
        ]

        score = np.mean(values)

        if score > best_score:

            best_score = score
            best_model = model_name
            best_aggregation = aggregation


print()
print("=" * 72)
print(
    "BEST DEVELOPMENT CONFIGURATION"
)
print("=" * 72)


print(
    "Model:       "
    + (
        "DINOv2"
        if best_model == "dino"
        else "MobileNet L8"
    )
)

print(
    "Aggregation: "
    + best_aggregation.upper()
)

print(
    f"Macro AUROC: "
    f"{best_score:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = Path(
    "artifacts/evaluation/"
    "mobilenet_vs_dino_dev_comparison.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    output,
    "w"
) as f:

    json.dump(
        {
            "protocol": {
                "split": "development_only",
                "categories": CATEGORIES,
                "aggregations": AGGREGATIONS,
                "final_test_used": False,
            },
            "results": results,
            "best_model": best_model,
            "best_aggregation": best_aggregation,
            "best_macro_auroc": float(
                best_score
            ),
        },
        f,
        indent=2
    )


print()
print(
    "Saved:",
    output
)