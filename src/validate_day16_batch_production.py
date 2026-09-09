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

from mobilenet_batch_extractor import (
    extract_batch,
    load_batch,
)


# ============================================================
# VISIONFORGE — DAY 16
# BATCHED PRODUCTION PATH VALIDATION
# ============================================================

NORMAL_SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

CALIBRATION_PATH = Path(
    "artifacts/evaluation/production_thresholds.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day16_batch_production_validation.json"
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
CHUNK_SIZE = 8192
BATCH_SIZE = 8

FUSION_WEIGHT_L4 = 0.5
FUSION_WEIGHT_L8 = 0.5

REVIEW_MULTIPLIER = 1.10

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

    model = mobilenet_v3_small(
        weights=weights,
    )

    model.eval()

    return model


# ============================================================
# SINGLE-IMAGE REFERENCE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_single(
    model,
    image_path,
):
    image_path = resolve_path(
        image_path
    )

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
            "Failed to capture L4/L8 features."
        )

    l4 = l4.squeeze(0)

    l4 = l4.permute(
        1,
        2,
        0,
    )

    l4 = l4.reshape(
        -1,
        l4.shape[-1],
    )

    l8 = l8.squeeze(0)

    l8 = l8.permute(
        1,
        2,
        0,
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

        l4, l8 = extract_single(
            model,
            path,
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
# CHUNKED PRODUCTION SCORER
# ============================================================

@torch.no_grad()
def score_chunked(
    query,
    reference_bank,
):
    best_similarity = torch.full(
        (query.shape[0],),
        -float("inf"),
        dtype=query.dtype,
        device=query.device,
    )

    num_references = reference_bank.shape[0]

    for start in range(
        0,
        num_references,
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            num_references,
        )

        reference_chunk = (
            reference_bank[start:end]
        )

        similarity = (
            query @ reference_chunk.T
        )

        chunk_best = (
            similarity.max(dim=1).values
        )

        best_similarity = torch.maximum(
            best_similarity,
            chunk_best,
        )

    squared_distance = (
        2.0 - 2.0 * best_similarity
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
# DECISION POLICY
# ============================================================

def get_p99_threshold(
    calibration,
    category,
):
    category_data = (
        calibration["categories"][category]
    )

    if "threshold" in category_data:
        return float(
            category_data["threshold"]
        )

    if "p99_threshold" in category_data:
        return float(
            category_data["p99_threshold"]
        )

    if "percentile_candidates" in category_data:

        for candidate in (
            category_data["percentile_candidates"]
        ):
            if (
                float(candidate["percentile"])
                == 99.0
            ):
                return float(
                    candidate["threshold"]
                )

    raise ValueError(
        f"Could not find a production "
        f"threshold for category: {category}"
    )


def classify(
    score,
    threshold,
):
    review_threshold = (
        threshold * REVIEW_MULTIPLIER
    )

    if score < threshold:
        decision = "PASS"

    elif score < review_threshold:
        decision = "REVIEW"

    else:
        decision = "FAIL"

    return (
        decision,
        review_threshold,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("VISIONFORGE — DAY 16")
    print("BATCHED PRODUCTION PATH VALIDATION")
    print("=" * 72)

    print()
    print("Architecture:")
    print("  MobileNetV3-Small")
    print("  L4 + L8")
    print("  Raw 50/50 fusion")
    print("  Euclidean-equivalent normalized scoring")
    print("  TOP5")
    print(f"  Chunk size = {CHUNK_SIZE}")
    print(f"  Batch size = {BATCH_SIZE}")

    split = load_json(
        NORMAL_SPLIT_PATH
    )

    calibration = load_json(
        CALIBRATION_PATH
    )

    model = load_model()

    results = {
        "architecture": (
            "MobileNetV3-Small L4+L8"
        ),
        "fusion": "raw_50_50",
        "distance": (
            "Euclidean-equivalent"
        ),
        "aggregation": "TOP5",
        "chunk_size": CHUNK_SIZE,
        "batch_size": BATCH_SIZE,
        "categories": {},
    }

    global_max_l4_difference = 0.0
    global_max_l8_difference = 0.0
    global_max_fusion_difference = 0.0

    decision_matches = 0
    decision_total = 0

    for category in CATEGORIES:

        print()
        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        reference_paths = (
            split["categories"][category][
                "reference"
            ]
        )

        development_paths = (
            split["categories"][category][
                "development"
            ]
        )

        test_paths = development_paths[:8]

        print(
            f"Reference images: "
            f"{len(reference_paths)}"
        )

        print(
            f"Validation images: "
            f"{len(test_paths)}"
        )

        print()
        print(
            "Building reference bank..."
        )

        l4_bank, l8_bank = (
            build_reference_bank(
                model,
                reference_paths,
            )
        )

        threshold = get_p99_threshold(
            calibration,
            category,
        )

        # ----------------------------------------------------
        # Single-image reference features
        # ----------------------------------------------------

        single_l4 = []
        single_l8 = []

        for path in test_paths:

            l4, l8 = extract_single(
                model,
                path,
            )

            single_l4.append(l4)
            single_l8.append(l8)

        # ----------------------------------------------------
        # Batched reference features
        # ----------------------------------------------------

        batch_l4_parts = []
        batch_l8_parts = []

        for start in range(
            0,
            len(test_paths),
            BATCH_SIZE,
        ):

            batch_paths = test_paths[
                start:start + BATCH_SIZE
            ]

            l4_batch, l8_batch = (
                extract_batch(
                    model,
                    [
                        resolve_path(path)
                        for path in batch_paths
                    ],
                )
            )

            batch_l4_parts.append(
                l4_batch
            )

            batch_l8_parts.append(
                l8_batch
            )

        batch_l4 = torch.cat(
            batch_l4_parts,
            dim=0,
        )

        batch_l8 = torch.cat(
            batch_l8_parts,
            dim=0,
        )

        category_results = {
            "threshold": threshold,
            "samples": [],
        }

        for index, path in enumerate(
            test_paths
        ):

            l4_original = single_l4[
                index
            ]

            l8_original = single_l8[
                index
            ]

            l4_batch = batch_l4[
                index
            ]

            l8_batch = batch_l8[
                index
            ]

            l4_difference = torch.max(
                torch.abs(
                    l4_original
                    - l4_batch
                )
            ).item()

            l8_difference = torch.max(
                torch.abs(
                    l8_original
                    - l8_batch
                )
            ).item()

            # ------------------------------------------------
            # Score both paths
            # ------------------------------------------------

            l4_original_score = (
                score_chunked(
                    l4_original,
                    l4_bank,
                )
            )

            l8_original_score = (
                score_chunked(
                    l8_original,
                    l8_bank,
                )
            )

            l4_batch_score = (
                score_chunked(
                    l4_batch,
                    l4_bank,
                )
            )

            l8_batch_score = (
                score_chunked(
                    l8_batch,
                    l8_bank,
                )
            )

            original_fusion = (
                FUSION_WEIGHT_L4
                * l4_original_score
                + FUSION_WEIGHT_L8
                * l8_original_score
            )

            batch_fusion = (
                FUSION_WEIGHT_L4
                * l4_batch_score
                + FUSION_WEIGHT_L8
                * l8_batch_score
            )

            fusion_difference = abs(
                (
                    original_fusion
                    - batch_fusion
                ).item()
            )

            original_decision, _ = (
                classify(
                    original_fusion.item(),
                    threshold,
                )
            )

            batch_decision, _ = (
                classify(
                    batch_fusion.item(),
                    threshold,
                )
            )

            decision_match = (
                original_decision
                == batch_decision
            )

            if decision_match:
                decision_matches += 1

            decision_total += 1

            global_max_l4_difference = max(
                global_max_l4_difference,
                l4_difference,
            )

            global_max_l8_difference = max(
                global_max_l8_difference,
                l8_difference,
            )

            global_max_fusion_difference = max(
                global_max_fusion_difference,
                fusion_difference,
            )

            print()
            print(path)
            print(
                "  L4 feature difference: "
                f"{l4_difference:.10f}"
            )
            print(
                "  L8 feature difference: "
                f"{l8_difference:.10f}"
            )
            print(
                "  Fusion difference: "
                f"{fusion_difference:.10f}"
            )
            print(
                "  Decision: "
                f"{batch_decision}"
            )
            print(
                "  Decision match: "
                f"{decision_match}"
            )

            category_results[
                "samples"
            ].append(
                {
                    "path": str(path),
                    "l4_difference": (
                        float(
                            l4_difference
                        )
                    ),
                    "l8_difference": (
                        float(
                            l8_difference
                        )
                    ),
                    "fusion_difference": (
                        float(
                            fusion_difference
                        )
                    ),
                    "original_fusion": (
                        float(
                            original_fusion.item()
                        )
                    ),
                    "batch_fusion": (
                        float(
                            batch_fusion.item()
                        )
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "original_decision": (
                        original_decision
                    ),
                    "batch_decision": (
                        batch_decision
                    ),
                    "decision_match": (
                        decision_match
                    ),
                }
            )

        results["categories"][
            category
        ] = category_results

    equivalence_pass = (
        global_max_l4_difference
        < 1e-5
        and global_max_l8_difference
        < 1e-5
        and global_max_fusion_difference
        < 1e-5
        and decision_matches
        == decision_total
    )

    results[
        "global_max_l4_difference"
    ] = float(
        global_max_l4_difference
    )

    results[
        "global_max_l8_difference"
    ] = float(
        global_max_l8_difference
    )

    results[
        "global_max_fusion_difference"
    ] = float(
        global_max_fusion_difference
    )

    results["decision_matches"] = int(
        decision_matches
    )

    results["decision_total"] = int(
        decision_total
    )

    results["equivalence_pass"] = bool(
        equivalence_pass
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
    print("FINAL VALIDATION")
    print("=" * 72)

    print(
        "L4 feature equivalence: "
        f"{global_max_l4_difference:.10f}"
    )

    print(
        "L8 feature equivalence: "
        f"{global_max_l8_difference:.10f}"
    )

    print(
        "Fusion equivalence: "
        f"{global_max_fusion_difference:.10f}"
    )

    print(
        f"Decision matches: "
        f"{decision_matches}/{decision_total}"
    )

    if equivalence_pass:
        print()
        print(
            "BATCHED PRODUCTION PATH: PASS"
        )
    else:
        print()
        print(
            "BATCHED PRODUCTION PATH: FAIL"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()