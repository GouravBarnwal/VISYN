import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import models, transforms


# ============================================================
# Configuration
# ============================================================

SEED = 42

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

DEFECT_SPLIT_PATH = Path(
    "artifacts/splits/defect_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day12_reference_bank_robustness.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

# Fractions of the canonical reference bank to retain.
REFERENCE_FRACTIONS = [
    1.0,
    0.75,
    0.50,
    0.25,
]

# Multiple deterministic subsets for each fraction.
N_REPEATS = 5

# Fixed aggregation from Day 11.
TOP_K = 5

# Fixed fusion from Day 11.
L4_WEIGHT = 0.50
L8_WEIGHT = 0.50


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# Model
# ============================================================

device = torch.device("cpu")

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

model = model.to(device)
model.eval()


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_l4_l8(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    x = tensor

    l4 = None
    l8 = None

    for idx, layer in enumerate(model.features):
        x = layer(x)

        if idx == 4:
            l4 = x.clone()

        if idx == 8:
            l8 = x.clone()

    if l4 is None or l8 is None:
        raise RuntimeError("Failed to capture L4/L8 features.")

    # Expected:
    # L4 -> [1, 40, 14, 14]
    # L8 -> [1, 48, 14, 14]

    l4 = l4.squeeze(0).permute(1, 2, 0).reshape(-1, l4.shape[1])
    l8 = l8.squeeze(0).permute(1, 2, 0).reshape(-1, l8.shape[1])

    l4 = torch.nn.functional.normalize(l4, dim=1)
    l8 = torch.nn.functional.normalize(l8, dim=1)

    return l4.cpu(), l8.cpu()


# ============================================================
# Path handling
# ============================================================

def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return Path("data") / path


# ============================================================
# Reference-bank scoring
# ============================================================

def score_against_bank(query_features, reference_features):
    """
    Calculate TOP-5 nearest-reference distance.

    For every query patch:
        nearest distance = minimum Euclidean distance
        image score = mean of TOP-5 largest patch distances
    """

    distances = torch.cdist(
        query_features,
        reference_features,
        p=2,
    )

    nearest = distances.min(dim=1).values

    k = min(TOP_K, nearest.numel())

    top_values = torch.topk(
        nearest,
        k=k,
        largest=True,
    ).values

    return float(top_values.mean().item())


def score_query(
    l4_query,
    l8_query,
    l4_bank,
    l8_bank,
):
    l4_score = score_against_bank(
        l4_query,
        l4_bank,
    )

    l8_score = score_against_bank(
        l8_query,
        l8_bank,
    )

    return l4_score, l8_score


# ============================================================
# Robust normalization
# ============================================================

def fit_normalization(normal_scores):
    """
    Fit median/MAD normalization using ONLY normal reference-bank
    development samples.

    This follows the Day 11 fusion protocol.
    """

    values = np.asarray(
        normal_scores,
        dtype=np.float64,
    )

    median = float(np.median(values))

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    # Avoid division by zero.
    if mad < 1e-8:
        mad = 1e-8

    return {
        "median": median,
        "mad": mad,
    }


def robust_normalize(value, normalization):
    return (
        value - normalization["median"]
    ) / (
        1.4826 * normalization["mad"]
    )


# ============================================================
# Load splits
# ============================================================

def load_splits():
    with open(
        NORMAL_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        normal_splits = json.load(f)

    with open(
        DEFECT_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        defect_splits = json.load(f)

    return normal_splits, defect_splits


# ============================================================
# Build feature cache
# ============================================================

def build_feature_cache(normal_splits, defect_splits):
    """
    Extract features once.

    This is important because the experiment changes only the
    reference bank. Query representations remain identical.
    """

    cache = {}

    for category in CATEGORIES:
        print(
            f"Extracting features: {category}"
        )

        normal_paths = (
            normal_splits["categories"]
            [category]["reference"]
        )

        normal_dev_paths = (
            normal_splits["categories"]
            [category]["development"]
        )

        defect_dev_paths = (
            defect_splits["categories"]
            [category]["development"]
        )

        all_paths = (
            list(normal_paths)
            + list(normal_dev_paths)
            + list(defect_dev_paths)
        )

        category_cache = {}

        for path_string in all_paths:
            if path_string in category_cache:
                continue

            path = resolve_path(path_string)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

            l4, l8 = extract_l4_l8(path)

            category_cache[path_string] = {
                "L4": l4,
                "L8": l8,
            }

        cache[category] = category_cache

    return cache


# ============================================================
# Evaluate one reference-bank configuration
# ============================================================

def evaluate_configuration(
    category,
    reference_paths,
    normal_dev_paths,
    defect_dev_paths,
    cache,
):
    """
    Evaluate:

        1. L8
        2. L4
        3. fixed 50/50 L4+L8 fusion

    Normalization is fitted from the normal development scores.
    """

    l4_bank = torch.cat(
        [
            cache[category][path]["L4"]
            for path in reference_paths
        ],
        dim=0,
    )

    l8_bank = torch.cat(
        [
            cache[category][path]["L8"]
            for path in reference_paths
        ],
        dim=0,
    )

    normal_l4 = []
    normal_l8 = []

    defect_l4 = []
    defect_l8 = []

    # --------------------------------------------------------
    # Normal development
    # --------------------------------------------------------

    for path in normal_dev_paths:
        l4_query = cache[category][path]["L4"]
        l8_query = cache[category][path]["L8"]

        l4_score, l8_score = score_query(
            l4_query,
            l8_query,
            l4_bank,
            l8_bank,
        )

        normal_l4.append(l4_score)
        normal_l8.append(l8_score)

    # --------------------------------------------------------
    # Defect development
    # --------------------------------------------------------

    for path in defect_dev_paths:
        l4_query = cache[category][path]["L4"]
        l8_query = cache[category][path]["L8"]

        l4_score, l8_score = score_query(
            l4_query,
            l8_query,
            l4_bank,
            l8_bank,
        )

        defect_l4.append(l4_score)
        defect_l8.append(l8_score)

    # --------------------------------------------------------
    # Fit normalization ONLY on normal development
    # --------------------------------------------------------

    l4_norm = fit_normalization(normal_l4)
    l8_norm = fit_normalization(normal_l8)

    normal_l4_norm = np.asarray(
        [
            robust_normalize(
                value,
                l4_norm,
            )
            for value in normal_l4
        ],
        dtype=np.float64,
    )

    normal_l8_norm = np.asarray(
        [
            robust_normalize(
                value,
                l8_norm,
            )
            for value in normal_l8
        ],
        dtype=np.float64,
    )

    defect_l4_norm = np.asarray(
        [
            robust_normalize(
                value,
                l4_norm,
            )
            for value in defect_l4
        ],
        dtype=np.float64,
    )

    defect_l8_norm = np.asarray(
        [
            robust_normalize(
                value,
                l8_norm,
            )
            for value in defect_l8
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Fixed fusion
    # --------------------------------------------------------

    normal_fusion = (
        L4_WEIGHT * normal_l4_norm
        + L8_WEIGHT * normal_l8_norm
    )

    defect_fusion = (
        L4_WEIGHT * defect_l4_norm
        + L8_WEIGHT * defect_l8_norm
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            np.zeros(len(normal_dev_paths)),
            np.ones(len(defect_dev_paths)),
        ]
    )

    scores = {
        "L4": np.concatenate(
            [
                normal_l4,
                defect_l4,
            ]
        ),
        "L8": np.concatenate(
            [
                normal_l8,
                defect_l8,
            ]
        ),
        "fusion_50_50": np.concatenate(
            [
                normal_fusion,
                defect_fusion,
            ]
        ),
    }

    metrics = {}

    for method, values in scores.items():
        metrics[method] = {
            "AUROC": float(
                roc_auc_score(
                    y_true,
                    values,
                )
            ),
            "AP": float(
                average_precision_score(
                    y_true,
                    values,
                )
            ),
        }

    return {
        "normal_count": len(normal_dev_paths),
        "defect_count": len(defect_dev_paths),
        "reference_count": len(reference_paths),
        "normalization": {
            "L4": l4_norm,
            "L8": l8_norm,
        },
        "metrics": metrics,
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    print("=" * 70)
    print("DAY 12 — REFERENCE-BANK ROBUSTNESS")
    print("=" * 70)

    normal_splits, defect_splits = load_splits()

    print("\nBuilding feature cache...")
    cache = build_feature_cache(
        normal_splits,
        defect_splits,
    )

    results = {
        "seed": SEED,
        "top_k": TOP_K,
        "fusion": {
            "L4_weight": L4_WEIGHT,
            "L8_weight": L8_WEIGHT,
        },
        "reference_fractions": REFERENCE_FRACTIONS,
        "repeats": N_REPEATS,
        "categories": {},
    }

    # --------------------------------------------------------
    # Per-category experiment
    # --------------------------------------------------------

    for category in CATEGORIES:
        print("\n" + "=" * 70)
        print(f"CATEGORY: {category}")
        print("=" * 70)

        reference_paths = list(
            normal_splits["categories"]
            [category]["reference"]
        )

        normal_dev_paths = list(
            normal_splits["categories"]
            [category]["development"]
        )

        defect_dev_paths = list(
            defect_splits["categories"]
            [category]["development"]
        )

        category_results = {
            "canonical_reference_count": len(
                reference_paths
            ),
            "runs": [],
        }

        for fraction in REFERENCE_FRACTIONS:
            if fraction == 1.0:
                subset_size = len(reference_paths)
            else:
                subset_size = max(
                    1,
                    int(
                        round(
                            len(reference_paths)
                            * fraction
                        )
                    ),
                )

            for repeat in range(N_REPEATS):
                run_seed = (
                    SEED
                    + int(fraction * 1000)
                    + repeat
                )

                rng = random.Random(run_seed)

                selected_paths = rng.sample(
                    reference_paths,
                    subset_size,
                )

                print(
                    f"  fraction={fraction:.2f} "
                    f"repeat={repeat + 1}/{N_REPEATS} "
                    f"references={subset_size}"
                )

                evaluation = evaluate_configuration(
                    category=category,
                    reference_paths=selected_paths,
                    normal_dev_paths=normal_dev_paths,
                    defect_dev_paths=defect_dev_paths,
                    cache=cache,
                )

                category_results["runs"].append(
                    {
                        "fraction": fraction,
                        "repeat": repeat + 1,
                        "seed": run_seed,
                        **evaluation,
                    }
                )

        results["categories"][category] = category_results

    # --------------------------------------------------------
    # Aggregate robustness statistics
    # --------------------------------------------------------

    for category in CATEGORIES:
        runs = results["categories"][category]["runs"]

        summary = {}

        for fraction in REFERENCE_FRACTIONS:
            fraction_runs = [
                run
                for run in runs
                if run["fraction"] == fraction
            ]

            fraction_summary = {}

            for method in [
                "L4",
                "L8",
                "fusion_50_50",
            ]:
                aurocs = np.asarray(
                    [
                        run["metrics"][method]["AUROC"]
                        for run in fraction_runs
                    ],
                    dtype=np.float64,
                )

                aps = np.asarray(
                    [
                        run["metrics"][method]["AP"]
                        for run in fraction_runs
                    ],
                    dtype=np.float64,
                )

                fraction_summary[method] = {
                    "AUROC_mean": float(
                        aurocs.mean()
                    ),
                    "AUROC_std": float(
                        aurocs.std(ddof=0)
                    ),
                    "AUROC_min": float(
                        aurocs.min()
                    ),
                    "AUROC_max": float(
                        aurocs.max()
                    ),
                    "AP_mean": float(
                        aps.mean()
                    ),
                    "AP_std": float(
                        aps.std(ddof=0)
                    ),
                    "AP_min": float(
                        aps.min()
                    ),
                    "AP_max": float(
                        aps.max()
                    ),
                }

            summary[str(fraction)] = (
                fraction_summary
            )

        results["categories"][category][
            "robustness_summary"
        ] = summary

    # --------------------------------------------------------
    # Macro summary
    # --------------------------------------------------------

    macro_summary = {}

    for fraction in REFERENCE_FRACTIONS:
        fraction_key = str(fraction)

        macro_summary[fraction_key] = {}

        for method in [
            "L4",
            "L8",
            "fusion_50_50",
        ]:
            category_means = []

            for category in CATEGORIES:
                value = results["categories"][
                    category
                ][
                    "robustness_summary"
                ][
                    fraction_key
                ][method]["AUROC_mean"]

                category_means.append(value)

            macro_summary[fraction_key][method] = {
                "macro_AUROC_mean": float(
                    np.mean(category_means)
                ),
                "category_AUROC_values": {
                    category: float(
                        results["categories"][
                            category
                        ][
                            "robustness_summary"
                        ][
                            fraction_key
                        ][method]["AUROC_mean"]
                    )
                    for category in CATEGORIES
                },
            }

    results["macro_summary"] = macro_summary

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DAY 12 REFERENCE-BANK ROBUSTNESS SUMMARY")
    print("=" * 70)

    for fraction in REFERENCE_FRACTIONS:
        key = str(fraction)

        print(
            f"\nReference fraction: "
            f"{fraction:.0%}"
        )

        for method in [
            "L4",
            "L8",
            "fusion_50_50",
        ]:
            value = macro_summary[key][method][
                "macro_AUROC_mean"
            ]

            print(
                f"  {method:15s}: "
                f"macro AUROC = {value:.4f}"
            )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nDAY 12 EXPERIMENT 1 COMPLETE")


if __name__ == "__main__":
    main()