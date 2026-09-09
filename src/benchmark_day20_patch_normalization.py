import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CATEGORIES = (
    "capsule",
    "screw",
)

IMAGE_SIZE = 224
TOP_K = 5

METHODS = (
    "RAW",
    "ZSCORE",
    "ROBUST",
    "PERCENTILE",
)

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    return model


def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        resolved = path
    else:
        resolved = DATA_ROOT / path

    if not resolved.exists():
        raise FileNotFoundError(
            f"Image does not exist: {resolved}"
        )

    return resolved


def preprocess_image(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return (tensor - MEAN) / STD


@torch.no_grad()
def extract_features(
    model,
    image_path,
):
    x = preprocess_image(
        image_path
    )

    l4 = None
    l8 = None

    for index, layer in enumerate(
        model.features
    ):
        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError(
            "Failed to capture L4/L8."
        )

    l4 = l4.permute(
        0,
        2,
        3,
        1,
    ).reshape(
        -1,
        l4.shape[1],
    )

    l8 = l8.permute(
        0,
        2,
        3,
        1,
    ).reshape(
        -1,
        l8.shape[1],
    )

    l4 = F.normalize(
        l4,
        p=2,
        dim=1,
    )

    l8 = F.normalize(
        l8,
        p=2,
        dim=1,
    )

    return l4, l8


def build_reference_bank(
    model,
    image_paths,
):
    l4_parts = []
    l8_parts = []

    for image_path in image_paths:
        l4, l8 = extract_features(
            model,
            image_path,
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

    return (
        torch.cat(
            l4_parts,
            dim=0,
        ),
        torch.cat(
            l8_parts,
            dim=0,
        ),
    )


def patch_scores(
    query,
    reference,
):
    similarity = query @ reference.T

    distance = torch.sqrt(
        torch.clamp(
            2.0 - 2.0 * similarity,
            min=0.0,
        )
    )

    return distance.min(
        dim=1
    ).values.cpu().numpy()


def image_patch_scores(
    model,
    image_paths,
    reference_l4,
    reference_l8,
):
    l4_results = []
    l8_results = []

    for image_path in image_paths:
        l4, l8 = extract_features(
            model,
            image_path,
        )

        l4_results.append(
            patch_scores(
                l4,
                reference_l4,
            )
        )

        l8_results.append(
            patch_scores(
                l8,
                reference_l8,
            )
        )

    return (
        l4_results,
        l8_results,
    )


def flatten(values):
    return np.concatenate(
        values
    )


def calculate_statistics(
    normal_patch_scores,
):
    values = flatten(
        normal_patch_scores
    )

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(values)
    )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    sorted_values = np.sort(
        values
    )

    return {
        "mean": mean,
        "std": max(std, 1e-8),
        "median": median,
        "mad": max(mad, 1e-8),
        "p01": float(
            np.percentile(
                sorted_values,
                1,
            )
        ),
        "p50": float(
            np.percentile(
                sorted_values,
                50,
            )
        ),
        "p95": float(
            np.percentile(
                sorted_values,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                sorted_values,
                99,
            )
        ),
    }


def normalize_patch_scores(
    scores,
    statistics,
    method,
):
    if method == "RAW":
        return scores

    if method == "ZSCORE":
        return (
            scores
            - statistics["mean"]
        ) / statistics["std"]

    if method == "ROBUST":
        return (
            scores
            - statistics["median"]
        ) / (
            1.4826
            * statistics["mad"]
        )

    if method == "PERCENTILE":
        low = statistics["p01"]
        high = statistics["p99"]

        normalized = (
            scores - low
        ) / max(
            high - low,
            1e-8,
        )

        return normalized

    raise ValueError(
        f"Unknown method: {method}"
    )


def aggregate(
    scores,
):
    top_values = np.sort(
        scores
    )[-TOP_K:]

    return float(
        np.mean(top_values)
    )


def score_images(
    l4_scores,
    l8_scores,
    l4_statistics,
    l8_statistics,
    method,
):
    results = []

    for l4, l8 in zip(
        l4_scores,
        l8_scores,
    ):
        normalized_l4 = (
            normalize_patch_scores(
                l4,
                l4_statistics,
                method,
            )
        )

        normalized_l8 = (
            normalize_patch_scores(
                l8,
                l8_statistics,
                method,
            )
        )

        l4_score = aggregate(
            normalized_l4
        )

        l8_score = aggregate(
            normalized_l8
        )

        fusion = (
            0.5 * l4_score
            + 0.5 * l8_score
        )

        results.append(
            fusion
        )

    return np.asarray(
        results,
        dtype=np.float64,
    )


def calculate_metrics(
    normal_scores,
    defect_scores,
):
    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_scores),
                dtype=np.int32,
            ),
            np.ones(
                len(defect_scores),
                dtype=np.int32,
            ),
        ]
    )

    y_scores = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    return {
        "auroc": float(
            roc_auc_score(
                y_true,
                y_scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_scores,
            )
        ),
    }


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "PATCH NORMALIZATION"
    )
    print("=" * 80)

    print(
        f"Categories: "
        f"{', '.join(CATEGORIES)}"
    )

    print(
        f"Resolution: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"TOP-K: {TOP_K}"
    )

    print()

    normal_path = (
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "normal_splits.json"
    )

    defect_path = (
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "defect_splits.json"
    )

    with open(
        normal_path,
        "r",
        encoding="utf-8",
    ) as file:
        normal_splits = json.load(file)

    with open(
        defect_path,
        "r",
        encoding="utf-8",
    ) as file:
        defect_splits = json.load(file)

    model = load_model()

    results = {}

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(
            f"CATEGORY: "
            f"{category.upper()}"
        )
        print("#" * 80)

        normal_category = (
            normal_splits[
                "categories"
            ][category]
        )

        defect_category = (
            defect_splits[
                "categories"
            ][category]
        )

        reference_images = [
            resolve_path(path)
            for path in normal_category[
                "reference"
            ]
        ]

        normal_images = [
            resolve_path(path)
            for path in normal_category[
                "development"
            ]
        ]

        defect_images = [
            resolve_path(path)
            for path in defect_category[
                "development"
            ]
        ]

        print(
            f"Reference: "
            f"{len(reference_images)}"
        )

        print(
            f"Normal development: "
            f"{len(normal_images)}"
        )

        print(
            f"Defect development: "
            f"{len(defect_images)}"
        )

        print(
            "Building reference banks..."
        )

        reference_l4, reference_l8 = (
            build_reference_bank(
                model,
                reference_images,
            )
        )

        print(
            "Extracting normal patch scores..."
        )

        normal_l4, normal_l8 = (
            image_patch_scores(
                model,
                normal_images,
                reference_l4,
                reference_l8,
            )
        )

        print(
            "Extracting defect patch scores..."
        )

        defect_l4, defect_l8 = (
            image_patch_scores(
                model,
                defect_images,
                reference_l4,
                reference_l8,
            )
        )

        l4_statistics = calculate_statistics(
            normal_l4
        )

        l8_statistics = calculate_statistics(
            normal_l8
        )

        print()
        print(
            "Normal patch statistics:"
        )

        print(
            "L4: "
            f"mean={l4_statistics['mean']:.4f} "
            f"std={l4_statistics['std']:.4f} "
            f"median={l4_statistics['median']:.4f} "
            f"MAD={l4_statistics['mad']:.4f}"
        )

        print(
            "L8: "
            f"mean={l8_statistics['mean']:.4f} "
            f"std={l8_statistics['std']:.4f} "
            f"median={l8_statistics['median']:.4f} "
            f"MAD={l8_statistics['mad']:.4f}"
        )

        category_results = {}

        for method in METHODS:
            normal_scores = score_images(
                normal_l4,
                normal_l8,
                l4_statistics,
                l8_statistics,
                method,
            )

            defect_scores = score_images(
                defect_l4,
                defect_l8,
                l4_statistics,
                l8_statistics,
                method,
            )

            threshold = float(
                np.percentile(
                    normal_scores,
                    99.0,
                )
            )

            false_positives = int(
                np.sum(
                    normal_scores
                    >= threshold
                )
            )

            detected = int(
                np.sum(
                    defect_scores
                    >= threshold
                )
            )

            metric = calculate_metrics(
                normal_scores,
                defect_scores,
            )

            category_results[
                method
            ] = {
                "threshold": threshold,
                "auroc": metric[
                    "auroc"
                ],
                "average_precision": metric[
                    "average_precision"
                ],
                "defect_detection": detected,
                "defect_count": len(
                    defect_scores
                ),
                "detection_rate": (
                    detected
                    / len(defect_scores)
                ),
                "normal_false_positives": (
                    false_positives
                ),
                "normal_count": len(
                    normal_scores
                ),
                "false_positive_rate": (
                    false_positives
                    / len(normal_scores)
                ),
                "normal_score_summary": {
                    "mean": float(
                        np.mean(
                            normal_scores
                        )
                    ),
                    "p95": float(
                        np.percentile(
                            normal_scores,
                            95,
                        )
                    ),
                    "max": float(
                        np.max(
                            normal_scores
                        )
                    ),
                },
                "defect_score_summary": {
                    "mean": float(
                        np.mean(
                            defect_scores
                        )
                    ),
                    "median": float(
                        np.median(
                            defect_scores
                        )
                    ),
                    "min": float(
                        np.min(
                            defect_scores
                        )
                    ),
                    "max": float(
                        np.max(
                            defect_scores
                        )
                    ),
                },
            }

            print(
                f"{method:<10} "
                f"AUROC={metric['auroc']:.4f} "
                f"AP={metric['average_precision']:.4f} "
                f"Detection="
                f"{detected}/"
                f"{len(defect_scores)} "
                f"("
                f"{detected / len(defect_scores) * 100:.1f}%"
                f") "
                f"FP="
                f"{false_positives}/"
                f"{len(normal_scores)}"
            )

        results[category] = {
            "normal_patch_statistics": {
                "L4": l4_statistics,
                "L8": l8_statistics,
            },
            "methods": category_results,
        }

    output = {
        "experiment": (
            "Day 20 patch-score "
            "normalization"
        ),
        "categories": list(
            CATEGORIES
        ),
        "image_size": IMAGE_SIZE,
        "top_k": TOP_K,
        "methods": list(
            METHODS
        ),
        "reference_split": (
            "normal_splits.json"
        ),
        "defect_split": (
            "defect_splits.json "
            "development only"
        ),
        "normal_statistics_source": (
            "normal development patch scores"
        ),
        "results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day20_patch_normalization.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print(
        f"Saved to: {output_path}"
    )
    print(
        "DAY 20 PATCH NORMALIZATION "
        "EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()