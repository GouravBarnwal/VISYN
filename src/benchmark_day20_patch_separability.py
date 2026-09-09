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
TOP_K_VALUES = (
    1,
    3,
    5,
    10,
    20,
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
    ).values


def get_patch_statistics(
    scores,
):
    values = scores.detach().cpu().numpy()

    return {
        "max": float(
            np.max(values)
        ),
        "top1": float(
            np.mean(
                np.sort(values)[-1:]
            )
        ),
        "top3": float(
            np.mean(
                np.sort(values)[-3:]
            )
        ),
        "top5": float(
            np.mean(
                np.sort(values)[-5:]
            )
        ),
        "top10": float(
            np.mean(
                np.sort(values)[-10:]
            )
        ),
        "top20": float(
            np.mean(
                np.sort(values)[-20:]
            )
        ),
        "mean": float(
            np.mean(values)
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
    }


def aggregate(
    scores,
    k,
):
    values = scores.detach().cpu().numpy()

    return float(
        np.mean(
            np.sort(values)[-k:]
        )
    )


def metrics(
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


def summarize_scores(
    scores,
):
    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),
        "max": float(
            np.max(values)
        ),
    }


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "PATCH-LEVEL SEPARABILITY"
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

        normal_l4 = []
        normal_l8 = []

        defect_l4 = []
        defect_l8 = []

        for image_path in normal_images:
            l4, l8 = extract_features(
                model,
                image_path,
            )

            normal_l4.append(
                get_patch_statistics(
                    patch_scores(
                        l4,
                        reference_l4,
                    )
                )
            )

            normal_l8.append(
                get_patch_statistics(
                    patch_scores(
                        l8,
                        reference_l8,
                    )
                )
            )

        for image_path in defect_images:
            l4, l8 = extract_features(
                model,
                image_path,
            )

            defect_l4.append(
                get_patch_statistics(
                    patch_scores(
                        l4,
                        reference_l4,
                    )
                )
            )

            defect_l8.append(
                get_patch_statistics(
                    patch_scores(
                        l8,
                        reference_l8,
                    )
                )
            )

        category_result = {}

        for layer_name, normal_data, defect_data in (
            (
                "L4",
                normal_l4,
                defect_l4,
            ),
            (
                "L8",
                normal_l8,
                defect_l8,
            ),
        ):
            print()
            print(
                f"--- {layer_name} ---"
            )

            layer_result = {
                "normal": {},
                "defect": {},
                "aggregation_metrics": {},
            }

            for statistic in (
                "max",
                "top1",
                "top3",
                "top5",
                "top10",
                "top20",
                "mean",
                "p95",
            ):
                normal_values = [
                    item[statistic]
                    for item in normal_data
                ]

                defect_values = [
                    item[statistic]
                    for item in defect_data
                ]

                layer_result["normal"][
                    statistic
                ] = summarize_scores(
                    normal_values
                )

                layer_result["defect"][
                    statistic
                ] = summarize_scores(
                    defect_values
                )

            for k in TOP_K_VALUES:
                normal_values = [
                    item[
                        f"top{k}"
                    ]
                    for item in normal_data
                ]

                defect_values = [
                    item[
                        f"top{k}"
                    ]
                    for item in defect_data
                ]

                layer_result[
                    "aggregation_metrics"
                ][
                    f"TOP{k}"
                ] = metrics(
                    np.asarray(
                        normal_values
                    ),
                    np.asarray(
                        defect_values
                    ),
                )

            category_result[
                layer_name
            ] = layer_result

            for statistic in (
                "max",
                "top1",
                "top3",
                "top5",
                "top10",
                "top20",
            ):
                normal_mean = np.mean(
                    [
                        item[statistic]
                        for item in normal_data
                    ]
                )

                defect_mean = np.mean(
                    [
                        item[statistic]
                        for item in defect_data
                    ]
                )

                print(
                    f"{statistic.upper():<6} "
                    f"normal={normal_mean:.4f} "
                    f"defect={defect_mean:.4f}"
                )

        results[category] = category_result

    output = {
        "experiment": (
            "Day 20 patch-level "
            "separability"
        ),
        "categories": list(
            CATEGORIES
        ),
        "image_size": IMAGE_SIZE,
        "top_k_values": list(
            TOP_K_VALUES
        ),
        "reference_split": (
            "normal_splits.json"
        ),
        "defect_split": (
            "defect_splits.json "
            "development only"
        ),
        "results": results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day20_patch_separability.json"
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
        "DAY 20 PATCH SEPARABILITY "
        "EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()