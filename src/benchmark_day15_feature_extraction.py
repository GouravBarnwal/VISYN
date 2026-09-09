import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


# ============================================================
# VISIONFORGE — DAY 15
# FEATURE EXTRACTION BENCHMARK
# ============================================================

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day15_feature_extraction.json"
)

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
# DATA
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
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

    model = mobilenet_v3_small(
        weights=weights
    )

    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    image_path = resolve_path(image_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array
    )

    tensor = tensor.permute(
        2, 0, 1
    )

    tensor = tensor.unsqueeze(0)

    tensor = (
        tensor - MEAN
    ) / STD

    return tensor


# ============================================================
# CURRENT EXTRACTION
# ============================================================

@torch.no_grad()
def extract_current(model, image):
    """
    Current implementation.

    Runs through all MobileNet feature blocks and captures
    L4 and L8.
    """

    x = image

    l4 = None
    l8 = None

    for index, layer in enumerate(
        model.features
    ):
        x = layer(x)

        if index == 4:
            l4 = x.clone()

        elif index == 8:
            l8 = x.clone()

    if l4 is None or l8 is None:
        raise RuntimeError(
            "Failed to capture L4/L8."
        )

    return l4, l8


# ============================================================
# OPTIMIZED EXTRACTION
# ============================================================

@torch.no_grad()
def extract_direct(model, image):
    """
    Optimized implementation.

    The model still performs exactly one forward pass.
    We avoid cloning intermediate tensors and capture the
    required tensors directly.
    """

    x = image

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

    return l4, l8


# ============================================================
# PATCH CONVERSION
# ============================================================

def convert_features(features):
    features = features.squeeze(0)

    features = features.permute(
        1, 2, 0
    )

    features = features.reshape(
        -1,
        features.shape[-1],
    )

    features = F.normalize(
        features,
        p=2,
        dim=1,
    )

    return features


# ============================================================
# EQUIVALENCE CHECK
# ============================================================

def compare_features(
    current,
    optimized,
):

    current_l4, current_l8 = current
    optimized_l4, optimized_l8 = optimized

    l4_difference = torch.max(
        torch.abs(
            current_l4
            - optimized_l4
        )
    ).item()

    l8_difference = torch.max(
        torch.abs(
            current_l8
            - optimized_l8
        )
    ).item()

    current_l4_patches = convert_features(
        current_l4
    )

    optimized_l4_patches = convert_features(
        optimized_l4
    )

    current_l8_patches = convert_features(
        current_l8
    )

    optimized_l8_patches = convert_features(
        optimized_l8
    )

    l4_patch_difference = torch.max(
        torch.abs(
            current_l4_patches
            - optimized_l4_patches
        )
    ).item()

    l8_patch_difference = torch.max(
        torch.abs(
            current_l8_patches
            - optimized_l8_patches
        )
    ).item()

    return {
        "raw_l4_max_difference":
            float(l4_difference),
        "raw_l8_max_difference":
            float(l8_difference),
        "normalized_l4_max_difference":
            float(l4_patch_difference),
        "normalized_l8_max_difference":
            float(l8_patch_difference),
    }


# ============================================================
# TIMING
# ============================================================

def benchmark(
    function,
    model,
    image,
):

    for _ in range(
        WARMUP_RUNS
    ):
        function(
            model,
            image,
        )

    times = []

    for _ in range(
        TIMED_RUNS
    ):
        start = time.perf_counter()

        function(
            model,
            image,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        times.append(
            elapsed
        )

    values = np.asarray(
        times,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(
            values.mean()
        ),
        "median_ms": float(
            np.median(values)
        ),
        "p95_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
        "min_ms": float(
            values.min()
        ),
        "max_ms": float(
            values.max()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — DAY 15"
    )
    print(
        "FEATURE EXTRACTION BENCHMARK"
    )
    print("=" * 72)

    print()
    print("Comparing:")
    print(
        "  Current: intermediate L4/L8 cloning"
    )
    print(
        "  Optimized: direct references"
    )

    print()
    print(
        "Both perform exactly one "
        "MobileNetV3-Small forward pass."
    )

    split = load_json(
        NORMAL_SPLIT_PATH
    )

    model = load_model()

    results = {
        "experiment": (
            "Day 15 feature extraction "
            "benchmark"
        ),
        "device": "CPU",
        "model": "MobileNetV3-Small",
        "layers": ["L4", "L8"],
        "categories": {},
    }

    global_max_difference = 0.0

    for category in CATEGORIES:

        print()
        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        query_paths = (
            split["categories"]
            [category]["development"]
        )

        benchmark_paths = query_paths[:10]

        print(
            f"Development images: "
            f"{len(query_paths)}"
        )

        print(
            f"Benchmark images: "
            f"{len(benchmark_paths)}"
        )

        category_result = {
            "query_count":
                len(benchmark_paths),
        }

        all_differences = []

        current_times = []
        optimized_times = []

        for path in benchmark_paths:

            image = preprocess_image(
                path
            )

            # ------------------------------------------------
            # Equivalence
            # ------------------------------------------------

            current_features = (
                extract_current(
                    model,
                    image,
                )
            )

            optimized_features = (
                extract_direct(
                    model,
                    image,
                )
            )

            comparison = compare_features(
                current_features,
                optimized_features,
            )

            all_differences.append(
                comparison
            )

            # ------------------------------------------------
            # Current timing
            # ------------------------------------------------

            current_timing = benchmark(
                extract_current,
                model,
                image,
            )

            current_times.append(
                current_timing
            )

            # ------------------------------------------------
            # Optimized timing
            # ------------------------------------------------

            optimized_timing = benchmark(
                extract_direct,
                model,
                image,
            )

            optimized_times.append(
                optimized_timing
            )

        max_raw_l4 = max(
            x["raw_l4_max_difference"]
            for x in all_differences
        )

        max_raw_l8 = max(
            x["raw_l8_max_difference"]
            for x in all_differences
        )

        max_norm_l4 = max(
            x["normalized_l4_max_difference"]
            for x in all_differences
        )

        max_norm_l8 = max(
            x["normalized_l8_max_difference"]
            for x in all_differences
        )

        global_max_difference = max(
            global_max_difference,
            max_raw_l4,
            max_raw_l8,
            max_norm_l4,
            max_norm_l8,
        )

        current_mean = float(
            np.mean(
                [
                    x["mean_ms"]
                    for x in current_times
                ]
            )
        )

        optimized_mean = float(
            np.mean(
                [
                    x["mean_ms"]
                    for x in optimized_times
                ]
            )
        )

        speedup = (
            current_mean
            / optimized_mean
        )

        reduction = (
            1.0
            - (
                optimized_mean
                / current_mean
            )
        ) * 100.0

        print()
        print(
            f"Max raw L4 difference: "
            f"{max_raw_l4:.10f}"
        )

        print(
            f"Max raw L8 difference: "
            f"{max_raw_l8:.10f}"
        )

        print(
            f"Max normalized L4 difference: "
            f"{max_norm_l4:.10f}"
        )

        print(
            f"Max normalized L8 difference: "
            f"{max_norm_l8:.10f}"
        )

        print()
        print(
            f"Current extraction: "
            f"{current_mean:.2f} ms"
        )

        print(
            f"Optimized extraction: "
            f"{optimized_mean:.2f} ms"
        )

        print(
            f"Speedup: "
            f"{speedup:.2f}x"
        )

        print(
            f"Latency reduction: "
            f"{reduction:.2f}%"
        )

        category_result[
            "equivalence"
        ] = {
            "raw_l4_max_difference":
                float(max_raw_l4),
            "raw_l8_max_difference":
                float(max_raw_l8),
            "normalized_l4_max_difference":
                float(max_norm_l4),
            "normalized_l8_max_difference":
                float(max_norm_l8),
        }

        category_result[
            "current_mean_ms"
        ] = current_mean

        category_result[
            "optimized_mean_ms"
        ] = optimized_mean

        category_result[
            "speedup"
        ] = float(speedup)

        category_result[
            "latency_reduction_percent"
        ] = float(reduction)

        results[
            "categories"
        ][category] = category_result

    results[
        "global_max_feature_difference"
    ] = float(
        global_max_difference
    )

    results[
        "equivalence_pass"
    ] = bool(
        global_max_difference < 1e-6
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
    print(
        "FINAL EQUIVALENCE CHECK"
    )
    print("=" * 72)

    print(
        f"Global maximum feature difference: "
        f"{global_max_difference:.10f}"
    )

    if results["equivalence_pass"]:
        print(
            "Feature equivalence: PASS"
        )
    else:
        print(
            "Feature equivalence: FAIL"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()