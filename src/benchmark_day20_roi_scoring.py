import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
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
CHUNK_SIZE = 8192

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
        return path

    return DATA_ROOT / path


def load_rgb_array(image_path):
    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def build_foreground_mask(image_array):
    """
    Estimate the object region from the image border.

    The border is treated as background.
    Pixels sufficiently different from the
    border appearance are considered foreground.
    """

    image = image_array.astype(
        np.float32
    )

    height, width, _ = image.shape

    border_size = max(
        4,
        min(height, width) // 20,
    )

    top = image[:border_size]
    bottom = image[
        height - border_size:
    ]
    left = image[
        :,
        :border_size,
    ]
    right = image[
        :,
        width - border_size:
    ]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )

    background_color = np.median(
        border_pixels,
        axis=0,
    )

    distance = np.linalg.norm(
        image - background_color,
        axis=2,
    )

    # Robust threshold based on border variation.
    border_distance = np.concatenate(
        [
            np.linalg.norm(
                top - background_color,
                axis=2,
            ).reshape(-1),
            np.linalg.norm(
                bottom - background_color,
                axis=2,
            ).reshape(-1),
            np.linalg.norm(
                left - background_color,
                axis=2,
            ).reshape(-1),
            np.linalg.norm(
                right - background_color,
                axis=2,
            ).reshape(-1),
        ]
    )

    median = np.median(
        border_distance
    )

    mad = np.median(
        np.abs(
            border_distance - median
        )
    )

    robust_scale = max(
        1.4826 * mad,
        3.0,
    )

    threshold = max(
        12.0,
        median + 4.0 * robust_scale,
    )

    mask = (
        distance > threshold
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Keep the largest connected foreground
    # component, which should correspond to
    # the inspected object.
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return np.ones(
            (height, width),
            dtype=np.uint8,
        )

    component_sizes = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_index = (
        1 + int(
            np.argmax(
                component_sizes
            )
        )
    )

    foreground = (
        labels == largest_index
    ).astype(
        np.uint8
    )

    # Avoid pathological tiny masks.
    foreground_fraction = (
        foreground.mean()
    )

    if foreground_fraction < 0.01:
        return np.ones(
            (height, width),
            dtype=np.uint8,
        )

    if foreground_fraction > 0.95:
        return np.ones(
            (height, width),
            dtype=np.uint8,
        )

    return foreground


def patch_indices_from_mask(
    mask,
    grid_height,
    grid_width,
):
    """
    Map an image-space foreground mask to
    feature-map patch centers.
    """

    image_height, image_width = mask.shape

    selected = []

    for row in range(grid_height):
        y = int(
            (
                row + 0.5
            )
            * image_height
            / grid_height
        )

        y = min(
            y,
            image_height - 1,
        )

        for col in range(grid_width):
            x = int(
                (
                    col + 0.5
                )
                * image_width
                / grid_width
            )

            x = min(
                x,
                image_width - 1,
            )

            if mask[y, x] > 0:
                selected.append(
                    row * grid_width
                    + col
                )

    if not selected:
        return np.arange(
            grid_height * grid_width
        )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


@torch.no_grad()
def extract_features(
    model,
    image_path,
):
    image_array = load_rgb_array(
        image_path
    )

    mask = build_foreground_mask(
        image_array
    )

    array = (
        image_array.astype(
            np.float32
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = tensor.unsqueeze(0)

    tensor = (
        tensor - MEAN
    ) / STD

    x = tensor

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

    l4_height = l4.shape[2]
    l4_width = l4.shape[3]

    l8_height = l8.shape[2]
    l8_width = l8.shape[3]

    l4_indices = (
        patch_indices_from_mask(
            mask,
            l4_height,
            l4_width,
        )
    )

    l8_indices = (
        patch_indices_from_mask(
            mask,
            l8_height,
            l8_width,
        )
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

    return (
        l4,
        l8,
        l4_indices,
        l8_indices,
    )


def build_reference_bank(
    model,
    image_paths,
    roi_only,
):
    l4_parts = []
    l8_parts = []

    for image_path in image_paths:
        (
            l4,
            l8,
            l4_indices,
            l8_indices,
        ) = extract_features(
            model,
            image_path,
        )

        if roi_only:
            l4 = l4[
                torch.from_numpy(
                    l4_indices
                )
            ]

            l8 = l8[
                torch.from_numpy(
                    l8_indices
                )
            ]

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


def score_layer(
    query,
    reference,
):
    best_distances = []

    for start in range(
        0,
        reference.shape[0],
        CHUNK_SIZE,
    ):
        reference_chunk = reference[
            start:start + CHUNK_SIZE
        ]

        similarity = (
            query @ reference_chunk.T
        )

        distance = torch.sqrt(
            torch.clamp(
                2.0 - 2.0 * similarity,
                min=0.0,
            )
        )

        best_distances.append(
            distance.min(
                dim=1
            ).values
        )

    patch_scores = torch.cat(
        best_distances,
        dim=0,
    )

    if patch_scores.numel() == 0:
        raise RuntimeError(
            "No patch scores available."
        )

    return torch.topk(
        patch_scores,
        k=min(
            TOP_K,
            patch_scores.numel(),
        ),
    ).values.mean().item()


def score_image(
    model,
    image_path,
    reference_l4,
    reference_l8,
    roi_only,
):
    (
        l4,
        l8,
        l4_indices,
        l8_indices,
    ) = extract_features(
        model,
        image_path,
    )

    if roi_only:
        l4 = l4[
            torch.from_numpy(
                l4_indices
            )
        ]

        l8 = l8[
            torch.from_numpy(
                l8_indices
            )
        ]

    l4_score = score_layer(
        l4,
        reference_l4,
    )

    l8_score = score_layer(
        l8,
        reference_l8,
    )

    fusion = (
        0.5 * l4_score
        + 0.5 * l8_score
    )

    return {
        "L4": l4_score,
        "L8": l8_score,
        "fusion": fusion,
        "roi_fraction_l4": (
            len(l4_indices)
            / 196.0
        ),
        "roi_fraction_l8": (
            len(l8_indices)
            / 196.0
        ),
    }


def evaluate(
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

    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_scores)
            ),
            np.ones(
                len(defect_scores)
            ),
        ]
    )

    y_scores = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    threshold = np.percentile(
        normal_scores,
        99.0,
    )

    detected = int(
        np.sum(
            defect_scores >= threshold
        )
    )

    false_positives = int(
        np.sum(
            normal_scores >= threshold
        )
    )

    return {
        "threshold": float(
            threshold
        ),
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
        "detected": detected,
        "defect_count": len(
            defect_scores
        ),
        "detection_rate": (
            detected
            / len(defect_scores)
        ),
        "false_positives": (
            false_positives
        ),
        "normal_count": len(
            normal_scores
        ),
        "false_positive_rate": (
            false_positives
            / len(normal_scores)
        ),
    }


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "ROI-ASSISTED SCORING"
    )
    print("=" * 80)

    print(
        f"Resolution: "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"TOP-K: {TOP_K}"
    )

    with open(
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "normal_splits.json",
        "r",
        encoding="utf-8",
    ) as file:
        normal_splits = json.load(file)

    with open(
        PROJECT_ROOT
        / "artifacts"
        / "splits"
        / "defect_splits.json",
        "r",
        encoding="utf-8",
    ) as file:
        defect_splits = json.load(file)

    model = load_model()

    all_results = {}

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(
            f"CATEGORY: {category.upper()}"
        )
        print("#" * 80)

        normal = normal_splits[
            "categories"
        ][category]

        defects = defect_splits[
            "categories"
        ][category]

        reference_images = [
            resolve_path(path)
            for path in normal[
                "reference"
            ]
        ]

        normal_images = [
            resolve_path(path)
            for path in normal[
                "development"
            ]
        ]

        defect_images = [
            resolve_path(path)
            for path in defects[
                "development"
            ]
        ]

        category_results = {}

        for mode, roi_only in (
            ("FULL_IMAGE", False),
            ("ROI_ONLY", True),
        ):
            print()
            print(
                f"--- {mode} ---"
            )

            print(
                "Building reference bank..."
            )

            reference_l4, reference_l8 = (
                build_reference_bank(
                    model,
                    reference_images,
                    roi_only,
                )
            )

            print(
                f"L4 bank: "
                f"{tuple(reference_l4.shape)}"
            )

            print(
                f"L8 bank: "
                f"{tuple(reference_l8.shape)}"
            )

            normal_scores = []
            normal_roi_l4 = []
            normal_roi_l8 = []

            print(
                "Scoring normal development..."
            )

            for path in normal_images:
                result = score_image(
                    model,
                    path,
                    reference_l4,
                    reference_l8,
                    roi_only,
                )

                normal_scores.append(
                    result["fusion"]
                )

                normal_roi_l4.append(
                    result["roi_fraction_l4"]
                )

                normal_roi_l8.append(
                    result["roi_fraction_l8"]
                )

            defect_scores = []
            defect_roi_l4 = []
            defect_roi_l8 = []

            print(
                "Scoring defect development..."
            )

            for path in defect_images:
                result = score_image(
                    model,
                    path,
                    reference_l4,
                    reference_l8,
                    roi_only,
                )

                defect_scores.append(
                    result["fusion"]
                )

                defect_roi_l4.append(
                    result["roi_fraction_l4"]
                )

                defect_roi_l8.append(
                    result["roi_fraction_l8"]
                )

            metrics = evaluate(
                normal_scores,
                defect_scores,
            )

            category_results[mode] = {
                **metrics,
                "normal_roi_fraction_l4_mean": (
                    float(
                        np.mean(
                            normal_roi_l4
                        )
                    )
                ),
                "normal_roi_fraction_l8_mean": (
                    float(
                        np.mean(
                            normal_roi_l8
                        )
                    )
                ),
                "defect_roi_fraction_l4_mean": (
                    float(
                        np.mean(
                            defect_roi_l4
                        )
                    )
                ),
                "defect_roi_fraction_l8_mean": (
                    float(
                        np.mean(
                            defect_roi_l8
                        )
                    )
                ),
            }

            print(
                f"AUROC="
                f"{metrics['auroc']:.4f} "
                f"AP="
                f"{metrics['average_precision']:.4f} "
                f"Detection="
                f"{metrics['detected']}/"
                f"{metrics['defect_count']} "
                f"("
                f"{metrics['detection_rate'] * 100:.1f}%"
                f") "
                f"FP="
                f"{metrics['false_positives']}/"
                f"{metrics['normal_count']}"
            )

            print(
                f"Mean ROI L4="
                f"{category_results[mode]['normal_roi_fraction_l4_mean']:.3f}"
            )

            print(
                f"Mean ROI L8="
                f"{category_results[mode]['normal_roi_fraction_l8_mean']:.3f}"
            )

        all_results[category] = (
            category_results
        )

    output = {
        "experiment": (
            "Day 20 ROI-assisted scoring"
        ),
        "resolution": IMAGE_SIZE,
        "top_k": TOP_K,
        "categories": list(
            CATEGORIES
        ),
        "reference_split": (
            "normal_splits.json"
        ),
        "defect_split": (
            "defect_splits.json "
            "development only"
        ),
        "results": all_results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day20_roi_scoring.json"
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
        "DAY 20 ROI EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()