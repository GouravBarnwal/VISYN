import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


CATEGORY = "screw"

RESOLUTIONS = [224, 320, 448]

IMAGES_PER_NORMAL = 20
IMAGES_PER_DEFECT = 10

DEFECT_TYPES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

P99 = 99.0
TOP_K = 5

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


def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (image_size, image_size),
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
    image_size,
):
    batch = preprocess_image(
        image_path,
        image_size,
    )

    x = batch
    l4 = None
    l8 = None

    for index, layer in enumerate(model.features):
        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError(
            "Failed to capture L4/L8 features."
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
    image_size,
):
    l4_parts = []
    l8_parts = []

    for image_path in image_paths:
        l4, l8 = extract_features(
            model,
            image_path,
            image_size,
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

    return (
        torch.cat(l4_parts, dim=0),
        torch.cat(l8_parts, dim=0),
    )


def top5_score(query, reference):
    similarity = query @ reference.T

    distance = torch.sqrt(
        torch.clamp(
            2.0 - 2.0 * similarity,
            min=0.0,
        )
    )

    patch_scores = distance.min(dim=1).values

    top_k = min(
        TOP_K,
        patch_scores.numel(),
    )

    return torch.topk(
        patch_scores,
        k=top_k,
    ).values.mean().item()


def score_image(
    model,
    image_path,
    image_size,
    reference_l4,
    reference_l8,
):
    l4, l8 = extract_features(
        model,
        image_path,
        image_size,
    )

    l4_score = top5_score(
        l4,
        reference_l4,
    )

    l8_score = top5_score(
        l8,
        reference_l8,
    )

    anomaly_score = (
        0.5 * l4_score
        + 0.5 * l8_score
    )

    return anomaly_score


def percentile_threshold(scores):
    return float(
        np.percentile(
            np.asarray(scores),
            P99,
        )
    )


def main():
    model = load_model()

    reference_images = sorted(
        (
            Path("data")
            / CATEGORY
            / "train"
            / "good"
        ).glob("*.png")
    )

    normal_images = sorted(
        (
            Path("data")
            / CATEGORY
            / "test"
            / "good"
        ).glob("*.png")
    )[:IMAGES_PER_NORMAL]

    if not reference_images:
        raise FileNotFoundError(
            "No screw training images found."
        )

    if not normal_images:
        raise FileNotFoundError(
            "No screw normal test images found."
        )

    print("=" * 72)
    print(
        "VISYN — MATCHED SCREW "
        "RESOLUTION EXPERIMENT"
    )
    print("=" * 72)
    print(
        f"Reference images: {len(reference_images)}"
    )
    print(
        f"Normal evaluation images: "
        f"{len(normal_images)}"
    )
    print()

    all_results = {}

    for image_size in RESOLUTIONS:
        print("=" * 72)
        print(
            f"RESOLUTION: "
            f"{image_size}x{image_size}"
        )
        print("=" * 72)

        print("Building matched reference bank...")

        start = time.perf_counter()

        reference_l4, reference_l8 = (
            build_reference_bank(
                model,
                reference_images,
                image_size,
            )
        )

        bank_time = (
            time.perf_counter()
            - start
        )

        print(
            f"Reference bank built in "
            f"{bank_time:.2f} seconds."
        )

        print()
        print("Scoring normal images...")

        normal_scores = []

        for image_path in normal_images:
            score = score_image(
                model,
                image_path,
                image_size,
                reference_l4,
                reference_l8,
            )

            normal_scores.append(score)

        threshold = percentile_threshold(
            normal_scores
        )

        print(
            f"P99 threshold: {threshold:.6f}"
        )

        print()
        print("Scoring defects...")

        defect_results = {}

        for defect_type in DEFECT_TYPES:
            defect_images = sorted(
                (
                    Path("data")
                    / CATEGORY
                    / "test"
                    / defect_type
                ).glob("*.png")
            )[:IMAGES_PER_DEFECT]

            scores = []

            for image_path in defect_images:
                score = score_image(
                    model,
                    image_path,
                    image_size,
                    reference_l4,
                    reference_l8,
                )

                scores.append(score)

            detected = sum(
                score >= threshold
                for score in scores
            )

            defect_results[defect_type] = {
                "count": len(scores),
                "detected": int(detected),
                "detection_rate": float(
                    detected / len(scores)
                ),
                "mean_score": float(
                    np.mean(scores)
                ),
                "min_score": float(
                    np.min(scores)
                ),
                "max_score": float(
                    np.max(scores)
                ),
            }

            print(
                f"{defect_type:20s} "
                f"detected="
                f"{detected}/{len(scores)} "
                f"mean="
                f"{np.mean(scores):.4f}"
            )

        normal_false_positives = sum(
            score >= threshold
            for score in normal_scores
        )

        print()
        print(
            f"Normal false positives: "
            f"{normal_false_positives}/"
            f"{len(normal_scores)}"
        )

        all_results[str(image_size)] = {
            "resolution": image_size,
            "reference_images": len(
                reference_images
            ),
            "normal_images": len(
                normal_images
            ),
            "threshold": threshold,
            "normal": {
                "mean_score": float(
                    np.mean(normal_scores)
                ),
                "median_score": float(
                    np.median(normal_scores)
                ),
                "p95_score": float(
                    np.percentile(
                        normal_scores,
                        95,
                    )
                ),
                "max_score": float(
                    np.max(normal_scores)
                ),
                "false_positives": int(
                    normal_false_positives
                ),
                "false_positive_rate": float(
                    normal_false_positives
                    / len(normal_scores)
                ),
            },
            "defects": defect_results,
            "reference_bank": {
                "L4_shape": list(
                    reference_l4.shape
                ),
                "L8_shape": list(
                    reference_l8.shape
                ),
            },
        }

        print()

    output = {
        "category": CATEGORY,
        "resolutions": RESOLUTIONS,
        "reference_split": "train_good",
        "normal_evaluation_split": (
            "test_good"
        ),
        "defect_types": DEFECT_TYPES,
        "top_k": TOP_K,
        "threshold_method": (
            "P99_normal_scores"
        ),
        "results": all_results,
    }

    output_path = Path(
        "artifacts"
        "/evaluation"
        "/day19_screw_matched_resolution.json"
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

    print("=" * 72)
    print(
        f"Saved to: {output_path}"
    )
    print(
        "MATCHED RESOLUTION EXPERIMENT: PASS"
    )


if __name__ == "__main__":
    main()