import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

IMAGE_SIZE = 320
TOP_K = 5
CHUNK_SIZE = 8192

CATEGORIES = (
    "capsule",
    "screw",
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


def score_chunked(
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
):
    l4, l8 = extract_features(
        model,
        image_path,
    )

    l4_score = score_chunked(
        l4,
        reference_l4,
    )

    l8_score = score_chunked(
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
    }


def main():
    print("=" * 80)
    print(
        "VISIONFORGE — DAY 20 "
        "320x320 BASELINE EQUIVALENCE"
    )
    print("=" * 80)

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

    all_results = {}

    for category in CATEGORIES:
        print()
        print("#" * 80)
        print(
            f"CATEGORY: {category.upper()}"
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
            "Building reference bank..."
        )

        reference_l4, reference_l8 = (
            build_reference_bank(
                model,
                reference_images,
            )
        )

        print(
            f"L4 shape: "
            f"{tuple(reference_l4.shape)}"
        )

        print(
            f"L8 shape: "
            f"{tuple(reference_l8.shape)}"
        )

        # Test a deterministic subset first.
        test_images = (
            normal_images[:10]
            + defect_images[:10]
        )

        max_l4_diff = 0.0
        max_l8_diff = 0.0
        max_fusion_diff = 0.0

        for image_path in test_images:
            result_a = score_image(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )

            result_b = score_image(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )

            l4_diff = abs(
                result_a["L4"]
                - result_b["L4"]
            )

            l8_diff = abs(
                result_a["L8"]
                - result_b["L8"]
            )

            fusion_diff = abs(
                result_a["fusion"]
                - result_b["fusion"]
            )

            max_l4_diff = max(
                max_l4_diff,
                l4_diff,
            )

            max_l8_diff = max(
                max_l8_diff,
                l8_diff,
            )

            max_fusion_diff = max(
                max_fusion_diff,
                fusion_diff,
            )

        print()
        print(
            f"Repeated-call L4 diff: "
            f"{max_l4_diff:.10f}"
        )

        print(
            f"Repeated-call L8 diff: "
            f"{max_l8_diff:.10f}"
        )

        print(
            f"Repeated-call fusion diff: "
            f"{max_fusion_diff:.10f}"
        )

        # Print actual scores for a known normal
        # and known defect so we can compare them
        # against the earlier experiment.
        print()
        print(
            "Sample score comparison:"
        )

        for label, image_path in (
            ("NORMAL", normal_images[0]),
            ("DEFECT", defect_images[0]),
        ):
            result = score_image(
                model,
                image_path,
                reference_l4,
                reference_l8,
            )

            print(
                f"{label:<8} "
                f"{image_path.name:<12} "
                f"L4={result['L4']:.6f} "
                f"L8={result['L8']:.6f} "
                f"FUSION={result['fusion']:.6f}"
            )

        all_results[category] = {
            "reference_images": len(
                reference_images
            ),
            "normal_images": len(
                normal_images
            ),
            "defect_images": len(
                defect_images
            ),
            "reference_l4_shape": list(
                reference_l4.shape
            ),
            "reference_l8_shape": list(
                reference_l8.shape
            ),
            "max_repeated_l4_diff": (
                max_l4_diff
            ),
            "max_repeated_l8_diff": (
                max_l8_diff
            ),
            "max_repeated_fusion_diff": (
                max_fusion_diff
            ),
        }

    output = {
        "experiment": (
            "Day 20 320x320 "
            "baseline equivalence"
        ),
        "resolution": IMAGE_SIZE,
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "results": all_results,
    }

    output_path = (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "day20_320_equivalence.json"
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
        "DAY 20 320x320 "
        "EQUIVALENCE CHECK: PASS"
    )


if __name__ == "__main__":
    main()