import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================================
# VISIONFORGE — DAY 16
# BATCHED FEATURE EXTRACTION BENCHMARK
# ============================================================

NORMAL_SPLIT_PATH = Path("artifacts/splits/normal_splits.json")

OUTPUT_PATH = Path("artifacts/evaluation/day16_batching.json")

DATA_ROOT = Path("data")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

IMAGE_SIZE = 224

BATCH_SIZES = [
    1,
    2,
    4,
    8,
]

WARMUP_RUNS = 3
TIMED_RUNS = 20

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


# ============================================================
# IO
# ============================================================


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def resolve_path(path):

    path = Path(path)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


# ============================================================
# MODEL
# ============================================================


def load_model():

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================


def preprocess_image(
    image_path,
):

    image_path = resolve_path(image_path)

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
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

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    tensor = (tensor - MEAN) / STD

    return tensor


def load_batch(
    paths,
):

    tensors = [preprocess_image(path) for path in paths]

    return torch.cat(
        tensors,
        dim=0,
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================


@torch.no_grad()
def extract_layers_batch(
    model,
    images,
):

    x = images

    l4 = None
    l8 = None

    for index, layer in enumerate(model.features):

        x = layer(x)

        if index == 4:
            l4 = x

        elif index == 8:
            l8 = x

    if l4 is None or l8 is None:
        raise RuntimeError("Failed to capture L4/L8 features.")

    # --------------------------------------------------------
    # [B, C, H, W]
    # ->
    # [B, H*W, C]
    # --------------------------------------------------------

    l4 = l4.permute(
        0,
        2,
        3,
        1,
    )

    l4 = l4.reshape(
        l4.shape[0],
        -1,
        l4.shape[-1],
    )

    l8 = l8.permute(
        0,
        2,
        3,
        1,
    )

    l8 = l8.reshape(
        l8.shape[0],
        -1,
        l8.shape[-1],
    )

    l4 = F.normalize(
        l4,
        p=2,
        dim=2,
    )

    l8 = F.normalize(
        l8,
        p=2,
        dim=2,
    )

    return l4, l8


@torch.no_grad()
def extract_single(
    model,
    image,
):

    l4, l8 = extract_layers_batch(
        model,
        image.unsqueeze(0),
    )

    return (
        l4[0],
        l8[0],
    )


# ============================================================
# TIMING
# ============================================================


def benchmark_batch_size(
    model,
    paths,
    batch_size,
):

    batches = []

    for start in range(
        0,
        len(paths),
        batch_size,
    ):

        batch_paths = paths[start : start + batch_size]

        batches.append(load_batch(batch_paths))

    for _ in range(WARMUP_RUNS):

        for batch in batches:
            extract_layers_batch(
                model,
                batch,
            )

    times = []

    for _ in range(TIMED_RUNS):

        start = time.perf_counter()

        for batch in batches:
            extract_layers_batch(
                model,
                batch,
            )

        elapsed = (time.perf_counter() - start) * 1000

        times.append(elapsed)

    values = np.asarray(
        times,
        dtype=np.float64,
    )

    total_images = len(paths)

    mean_total_ms = float(values.mean())

    mean_per_image_ms = mean_total_ms / total_images

    throughput = total_images / (mean_total_ms / 1000.0)

    return {
        "batch_size": batch_size,
        "mean_total_ms": mean_total_ms,
        "median_total_ms": float(np.median(values)),
        "p95_total_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
        "mean_per_image_ms": mean_per_image_ms,
        "images_per_second": float(throughput),
    }


# ============================================================
# FEATURE EQUIVALENCE
# ============================================================


@torch.no_grad()
def check_equivalence(
    model,
    paths,
):

    images = load_batch(paths)

    batched_l4, batched_l8 = extract_layers_batch(
        model,
        images,
    )

    max_l4_difference = 0.0
    max_l8_difference = 0.0

    for index in range(len(paths)):

        single_l4, single_l8 = extract_single(
            model,
            images[index],
        )

        l4_difference = torch.max(torch.abs(single_l4 - batched_l4[index])).item()

        l8_difference = torch.max(torch.abs(single_l8 - batched_l8[index])).item()

        max_l4_difference = max(
            max_l4_difference,
            l4_difference,
        )

        max_l8_difference = max(
            max_l8_difference,
            l8_difference,
        )

    return {
        "max_l4_feature_difference": float(max_l4_difference),
        "max_l8_feature_difference": float(max_l8_difference),
        "pass": (max_l4_difference < 1e-5 and max_l8_difference < 1e-5),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 72)
    print("VISIONFORGE — DAY 16")
    print("BATCHED FEATURE EXTRACTION BENCHMARK")
    print("=" * 72)

    split = load_json(NORMAL_SPLIT_PATH)

    model = load_model()

    results = {
        "experiment": "Day 16 batched feature extraction",
        "device": "CPU",
        "batch_sizes": BATCH_SIZES,
        "categories": {},
    }

    global_l4_difference = 0.0
    global_l8_difference = 0.0

    for category in CATEGORIES:

        print()
        print("=" * 72)
        print(f"CATEGORY: {category}")
        print("=" * 72)

        paths = split["categories"][category]["development"]

        # Keep the benchmark identical in image count
        # across batch sizes.
        paths = paths[:16]

        print(f"Benchmark images: " f"{len(paths)}")

        equivalence = check_equivalence(
            model,
            paths[:4],
        )

        print()
        print("Feature equivalence:")

        print(
            f"  L4 max difference: " f"{equivalence['max_l4_feature_difference']:.10f}"
        )

        print(
            f"  L8 max difference: " f"{equivalence['max_l8_feature_difference']:.10f}"
        )

        print(f"  Result: " f"{'PASS' if equivalence['pass'] else 'FAIL'}")

        global_l4_difference = max(
            global_l4_difference,
            equivalence["max_l4_feature_difference"],
        )

        global_l8_difference = max(
            global_l8_difference,
            equivalence["max_l8_feature_difference"],
        )

        category_results = {
            "benchmark_images": len(paths),
            "feature_equivalence": equivalence,
            "batch_results": {},
        }

        for batch_size in BATCH_SIZES:

            result = benchmark_batch_size(
                model,
                paths,
                batch_size,
            )

            category_results["batch_results"][str(batch_size)] = result

            print()
            print(f"Batch size {batch_size}:")

            print(f"  Mean total: " f"{result['mean_total_ms']:.3f} ms")

            print(f"  Mean per image: " f"{result['mean_per_image_ms']:.3f} ms")

            print(f"  Throughput: " f"{result['images_per_second']:.2f} images/s")

        results["categories"][category] = category_results

    results["global_max_l4_feature_difference"] = float(global_l4_difference)

    results["global_max_l8_feature_difference"] = float(global_l8_difference)

    results["feature_equivalence_pass"] = bool(
        global_l4_difference < 1e-5 and global_l8_difference < 1e-5
    )

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
    print("FINAL RESULT")
    print("=" * 72)

    print(f"Global L4 feature difference: " f"{global_l4_difference:.10f}")

    print(f"Global L8 feature difference: " f"{global_l8_difference:.10f}")

    if results["feature_equivalence_pass"]:

        print("Feature equivalence: PASS")

    else:

        print("Feature equivalence: FAIL")

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
