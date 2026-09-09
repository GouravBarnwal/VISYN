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

AGGREGATIONS = (
    "TOP1",
    "TOP3",
    "TOP5",
    "TOP10",
    "TOP20",
)

IMAGE_SIZE = 224

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
    ).values


def aggregate(
    scores,
    method,
):
    if method == "TOP1":
        k = 1

    elif method == "TOP3":
        k = 3

    elif method == "TOP5":
        k = 5

    elif method == "TOP10":
        k = 10

    elif method == "TOP20":
        k = 20

    else:
        raise ValueError(
            f"Unknown aggregation: {method}"
        )

    k = min(
        k,
        scores.numel(),
    )

    return torch.topk(
        scores,
        k=k,
    ).values.mean().item()


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


def score_image(
    model,
    image_path,
    reference_l4,
    reference_l8,
    method,
):
    l4, l8 = extract_features(
        model,
        image_path,
    )

    l4_patch_scores = patch_scores(
        l4,
        reference_l4,
    )

    l8_patch_scores = patch_scores(
        l8,
        reference_l8,
    )

    l4_score = aggregate(
        l4_patch_scores,
        method,
    )

    l8_score = aggregate(
        l8_patch_scores,
        method,
    )

    fusion = (
        0.5 * l4_score
        + 0.5 * l8_score
    )

    return fusion


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


def defect_type_from_path(
    image_path,
):
    return image_path.parent.name


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "AGGREGATION TUNING"
    )
    print("=" * 80)
    print(
        f"Categories: "
        f"{', '.join(CATEGORIES)}"
    )
    print(
        f"Image size: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
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
            f"Reference images: "
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

        reference_l4, reference_l8 = (
            build_reference_bank(
                model,
                reference_images,
            )
        )

        category_results = {}

        for method in AGGREGATIONS:
            print()
            print(
                f"--- {method} ---"
            )

            normal_scores = np.asarray(
                [
                    score_image(
                        model,
                        image_path,
                        reference_l4,
                        reference_l8,
                        method,
                    )
                    for image_path in normal_images
                ],
                dtype=np.float64,
            )

            threshold = float(
                np.percentile(
                    normal_scores,
                    99.0,
                )
            )

            defect_scores = np.asarray(
                [
                    score_image(
                        model,
                        image_path,
                        reference_l4,
                        reference_l8,
                        method,
                    )
                    for image_path in defect_images
                ],
                dtype=np.float64,
            )

            metrics = calculate_metrics(
                normal_scores,
                defect_scores,
            )

            false_positives = int(
                np.sum(
                    normal_scores >= threshold
                )
            )

            detected = int(
                np.sum(
                    defect_scores >= threshold
                )
            )

            type_results = {}

            for defect_type in sorted(
                {
                    defect_type_from_path(
                        path
                    )
                    for path in defect_images
                }
            ):
                type_scores = []

                for index, image_path in enumerate(
                    defect_images
                ):
                    if (
                        defect_type_from_path(
                            image_path
                        )
                        == defect_type
                    ):
                        type_scores.append(
                            defect_scores[
                                index
                            ]
                        )

                type_scores = np.asarray(
                    type_scores,
                    dtype=np.float64,
                )

                type_detected = int(
                    np.sum(
                        type_scores
                        >= threshold
                    )
                )

                type_results[
                    defect_type
                ] = {
                    "count": len(
                        type_scores
                    ),
                    "detected": type_detected,
                    "detection_rate": (
                        type_detected
                        / len(type_scores)
                    ),
                    "mean_score": float(
                        np.mean(type_scores)
                    ),
                }

            category_results[
                method
            ] = {
                "threshold": threshold,
                "normal": {
                    "count": len(
                        normal_scores
                    ),
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
                    "false_positives": (
                        false_positives
                    ),
                    "false_positive_rate": (
                        false_positives
                        / len(normal_scores)
                    ),
                },
                "defects": {
                    "count": len(
                        defect_scores
                    ),
                    "detected": detected,
                    "detection_rate": (
                        detected
                        / len(defect_scores)
                    ),
                    "mean": float(
                        np.mean(
                            defect_scores
                        )
                    ),
                },
                "metrics": metrics,
                "defect_types": type_results,
            }

            print(
                f"Threshold: "
                f"{threshold:.6f}"
            )

            print(
                f"AUROC: "
                f"{metrics['auroc']:.4f}"
            )

            print(
                f"AP: "
                f"{metrics['average_precision']:.4f}"
            )

            print(
                f"Detection: "
                f"{detected}/"
                f"{len(defect_scores)} "
                f"("
                f"{detected / len(defect_scores) * 100:.1f}%"
                f")"
            )

            print(
                f"Normal FP: "
                f"{false_positives}/"
                f"{len(normal_scores)}"
            )

            print(
                "Per-defect detection:"
            )

            for defect_type, values in (
                type_results.items()
            ):
                print(
                    f"  {defect_type}: "
                    f"{values['detected']}/"
                    f"{values['count']} "
                    f"("
                    f"{values['detection_rate'] * 100:.1f}%"
                    f")"
                )

        results[category] = (
            category_results
        )

    output = {
        "experiment": (
            "Day 20 aggregation tuning"
        ),
        "categories": list(
            CATEGORIES
        ),
        "image_size": IMAGE_SIZE,
        "aggregations": list(
            AGGREGATIONS
        ),
        "reference_split": (
            "normal_splits.json"
        ),
        "defect_split": (
            "defect_splits.json "
            "development only"
        ),
        "fusion": (
            "0.5 L4 + 0.5 L8"
        ),
        "results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day20_aggregation_tuning.json"
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
        "DAY 20 AGGREGATION "
        "EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()