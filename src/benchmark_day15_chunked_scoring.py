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
# CHUNKED SCORING TRADE-OFF BENCHMARK
# ============================================================

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day15_chunked_tradeoff.json"
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
TOP_K = 5

WARMUP_RUNS = 3
TIMED_RUNS = 30

CHUNK_SIZES = [
    2048,
    4096,
    8192,
    16384,
]

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
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_layers(model, image_tensor):

    x = image_tensor

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
            "Failed to capture L4/L8 features."
        )

    l4 = l4.squeeze(0)
    l4 = l4.permute(
        1, 2, 0
    )
    l4 = l4.reshape(
        -1,
        l4.shape[-1],
    )

    l8 = l8.squeeze(0)
    l8 = l8.permute(
        1, 2, 0
    )
    l8 = l8.reshape(
        -1,
        l8.shape[-1],
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


# ============================================================
# REFERENCE BANK
# ============================================================

@torch.no_grad()
def build_reference_bank(
    model,
    reference_paths,
):

    l4_parts = []
    l8_parts = []

    for path in reference_paths:

        image = preprocess_image(
            path
        )

        l4, l8 = extract_layers(
            model,
            image,
        )

        l4_parts.append(l4)
        l8_parts.append(l8)

    l4_bank = torch.cat(
        l4_parts,
        dim=0,
    )

    l8_bank = torch.cat(
        l8_parts,
        dim=0,
    )

    return l4_bank, l8_bank


# ============================================================
# SCORING
# ============================================================

@torch.no_grad()
def score_full(
    query,
    reference_bank,
):
    similarity = (
        query @ reference_bank.T
    )

    nearest_similarity = similarity.max(
        dim=1
    ).values

    squared_distance = (
        2.0
        - 2.0 * nearest_similarity
    )

    squared_distance = torch.clamp(
        squared_distance,
        min=0.0,
    )

    distances = torch.sqrt(
        squared_distance
    )

    topk = torch.topk(
        distances,
        k=TOP_K,
        largest=True,
    ).values

    return topk.mean()


@torch.no_grad()
def score_chunked(
    query,
    reference_bank,
    chunk_size,
):
    num_references = reference_bank.shape[0]

    best_similarity = torch.full(
        (query.shape[0],),
        -float("inf"),
        dtype=query.dtype,
        device=query.device,
    )

    for start in range(
        0,
        num_references,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            num_references,
        )

        reference_chunk = (
            reference_bank[start:end]
        )

        similarity = (
            query @ reference_chunk.T
        )

        chunk_best = similarity.max(
            dim=1
        ).values

        best_similarity = torch.maximum(
            best_similarity,
            chunk_best,
        )

    squared_distance = (
        2.0
        - 2.0 * best_similarity
    )

    squared_distance = torch.clamp(
        squared_distance,
        min=0.0,
    )

    distances = torch.sqrt(
        squared_distance
    )

    topk = torch.topk(
        distances,
        k=TOP_K,
        largest=True,
    ).values

    return topk.mean()


# ============================================================
# TIMING
# ============================================================

def time_full(
    query,
    bank,
):
    for _ in range(WARMUP_RUNS):
        score_full(
            query,
            bank,
        )

    times = []

    for _ in range(TIMED_RUNS):

        start = time.perf_counter()

        score_full(
            query,
            bank,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        times.append(
            elapsed
        )

    return summarize_times(
        times
    )


def time_chunked(
    query,
    bank,
    chunk_size,
):
    for _ in range(WARMUP_RUNS):
        score_chunked(
            query,
            bank,
            chunk_size,
        )

    times = []

    for _ in range(TIMED_RUNS):

        start = time.perf_counter()

        score_chunked(
            query,
            bank,
            chunk_size,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        times.append(
            elapsed
        )

    return summarize_times(
        times
    )


def summarize_times(times):

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
    }


# ============================================================
# MEMORY CALCULATION
# ============================================================

def similarity_matrix_memory_mb(
    query_patches,
    reference_count,
):
    bytes_used = (
        query_patches
        * reference_count
        * 4
    )

    return bytes_used / (
        1024 * 1024
    )


def chunk_similarity_memory_mb(
    query_patches,
    chunk_size,
):
    bytes_used = (
        query_patches
        * chunk_size
        * 4
    )

    return bytes_used / (
        1024 * 1024
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — DAY 15"
    )
    print(
        "CHUNKED SCORING TRADE-OFF"
    )
    print("=" * 72)

    print()
    print(
        "Full scoring creates the complete"
    )
    print(
        "query × reference similarity matrix."
    )

    print()
    print(
        "Chunked scoring processes the reference"
    )
    print(
        "bank in smaller blocks."
    )

    split = load_json(
        NORMAL_SPLIT_PATH
    )

    model = load_model()

    results = {
        "experiment":
            "Day 15 chunked scoring trade-off",
        "device":
            "CPU",
        "top_k":
            TOP_K,
        "chunk_sizes":
            CHUNK_SIZES,
        "categories":
            {},
    }

    global_max_difference = 0.0

    for category in CATEGORIES:

        print()
        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        reference_paths = (
            split["categories"]
            [category]["reference"]
        )

        query_paths = (
            split["categories"]
            [category]["development"]
        )

        benchmark_path = query_paths[0]

        print(
            f"Reference images: "
            f"{len(reference_paths)}"
        )

        print(
            f"Benchmark image: "
            f"{benchmark_path}"
        )

        print()
        print(
            "Building reference banks..."
        )

        l4_bank, l8_bank = (
            build_reference_bank(
                model,
                reference_paths,
            )
        )

        image = preprocess_image(
            benchmark_path
        )

        l4_query, l8_query = (
            extract_layers(
                model,
                image,
            )
        )

        category_results = {}

        for layer_name, query, bank in [
            (
                "L4",
                l4_query,
                l4_bank,
            ),
            (
                "L8",
                l8_query,
                l8_bank,
            ),
        ]:

            query_patches = query.shape[0]
            reference_count = bank.shape[0]

            full_memory = (
                similarity_matrix_memory_mb(
                    query_patches,
                    reference_count,
                )
            )

            print()
            print(
                f"--- {layer_name} ---"
            )

            print(
                f"Query patches: "
                f"{query_patches}"
            )

            print(
                f"Reference vectors: "
                f"{reference_count}"
            )

            print(
                f"Full similarity matrix: "
                f"{full_memory:.2f} MB"
            )

            baseline_score = score_full(
                query,
                bank,
            )

            baseline_timing = time_full(
                query,
                bank,
            )

            print(
                f"Full mean latency: "
                f"{baseline_timing['mean_ms']:.3f} ms"
            )

            layer_results = {
                "query_patches":
                    int(query_patches),
                "reference_vectors":
                    int(reference_count),
                "full_similarity_matrix_mb":
                    float(full_memory),
                "full_baseline":
                    {
                        "score":
                            float(
                                baseline_score.item()
                            ),
                        "timing_ms":
                            baseline_timing,
                    },
                "chunks":
                    {},
            }

            for chunk_size in CHUNK_SIZES:

                chunk_memory = (
                    chunk_similarity_memory_mb(
                        query_patches,
                        chunk_size,
                    )
                )

                candidate_score = (
                    score_chunked(
                        query,
                        bank,
                        chunk_size,
                    )
                )

                difference = abs(
                    (
                        candidate_score
                        - baseline_score
                    ).item()
                )

                global_max_difference = max(
                    global_max_difference,
                    difference,
                )

                timing = time_chunked(
                    query,
                    bank,
                    chunk_size,
                )

                speed_ratio = (
                    baseline_timing["mean_ms"]
                    / timing["mean_ms"]
                )

                memory_reduction = (
                    1.0
                    - (
                        chunk_memory
                        / full_memory
                    )
                ) * 100.0

                print()
                print(
                    f"Chunk {chunk_size}:"
                )

                print(
                    f"  Score difference: "
                    f"{difference:.10f}"
                )

                print(
                    f"  Similarity workspace: "
                    f"{chunk_memory:.2f} MB"
                )

                print(
                    f"  Memory reduction: "
                    f"{memory_reduction:.1f}%"
                )

                print(
                    f"  Mean latency: "
                    f"{timing['mean_ms']:.3f} ms"
                )

                print(
                    f"  Relative speed: "
                    f"{speed_ratio:.2f}x"
                )

                layer_results[
                    "chunks"
                ][str(chunk_size)] = {
                    "score":
                        float(
                            candidate_score.item()
                        ),
                    "score_difference":
                        float(difference),
                    "similarity_workspace_mb":
                        float(chunk_memory),
                    "memory_reduction_percent":
                        float(memory_reduction),
                    "timing_ms":
                        timing,
                    "relative_speed_vs_full":
                        float(speed_ratio),
                }

            category_results[
                layer_name
            ] = layer_results

        results[
            "categories"
        ][category] = category_results

    results[
        "global_max_score_difference"
    ] = float(
        global_max_difference
    )

    results[
        "equivalence_pass"
    ] = bool(
        global_max_difference < 1e-5
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
        "FINAL RESULT"
    )
    print("=" * 72)

    print(
        f"Global maximum score difference: "
        f"{global_max_difference:.10f}"
    )

    if results["equivalence_pass"]:
        print(
            "Numerical equivalence: PASS"
        )
    else:
        print(
            "Numerical equivalence: FAIL"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()