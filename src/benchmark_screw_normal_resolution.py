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

IMAGES_TO_TEST = 20

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)

THRESHOLD = 0.6448165547847747


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

    tensor = (tensor - MEAN) / STD

    return tensor


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


def load_reference_banks():
    root = Path(
        "artifacts"
        "/production"
        "/reference_banks"
        "/screw"
    )

    l4 = torch.from_numpy(
        np.load(root / "L4.npy")
    ).float()

    l8 = torch.from_numpy(
        np.load(root / "L8.npy")
    ).float()

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
        5,
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


def main():
    model = load_model()

    reference_l4, reference_l8 = (
        load_reference_banks()
    )

    images = sorted(
        (
            Path("data")
            / CATEGORY
            / "test"
            / "good"
        ).glob("*.png")
    )[:IMAGES_TO_TEST]

    if not images:
        raise FileNotFoundError(
            "No normal screw images found."
        )

    print("=" * 72)
    print(
        "VISIONFORGE — NORMAL SCREW "
        "RESOLUTION ANALYSIS"
    )
    print("=" * 72)

    print(
        f"Testing {len(images)} normal screw images."
    )
    print(
        f"Production threshold: {THRESHOLD:.6f}"
    )
    print()

    results = {}

    for image_size in RESOLUTIONS:
        scores = []

        start = time.perf_counter()

        for image_path in images:
            score = score_image(
                model,
                image_path,
                image_size,
                reference_l4,
                reference_l8,
            )

            scores.append(score)

        elapsed = (
            time.perf_counter()
            - start
        )

        scores = np.asarray(scores)

        false_positives = int(
            np.sum(scores >= THRESHOLD)
        )

        results[str(image_size)] = {
            "count": len(scores),
            "mean_score": float(
                np.mean(scores)
            ),
            "median_score": float(
                np.median(scores)
            ),
            "max_score": float(
                np.max(scores)
            ),
            "p95_score": float(
                np.percentile(scores, 95)
            ),
            "false_positives": false_positives,
            "false_positive_rate": float(
                false_positives / len(scores)
            ),
            "images_per_second": float(
                len(images) / elapsed
            ),
        }

        print(
            f"{image_size}x{image_size}"
        )
        print(
            f"  Mean:       {np.mean(scores):.4f}"
        )
        print(
            f"  Median:     {np.median(scores):.4f}"
        )
        print(
            f"  P95:        {np.percentile(scores, 95):.4f}"
        )
        print(
            f"  Max:        {np.max(scores):.4f}"
        )
        print(
            f"  False +:    "
            f"{false_positives}/{len(scores)}"
        )
        print()

    output = {
        "category": CATEGORY,
        "images_tested": len(images),
        "threshold": THRESHOLD,
        "resolutions": RESOLUTIONS,
        "results": results,
    }

    output_path = Path(
        "artifacts"
        "/evaluation"
        "/day19_screw_normal_resolution.json"
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
        "NORMAL SCREW RESOLUTION ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()