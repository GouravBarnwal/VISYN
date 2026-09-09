import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================================
# VISYN — DAY 15
# OPTIMIZED PRODUCTION PATH VALIDATION
# ============================================================

NORMAL_SPLIT_PATH = Path("artifacts/splits/normal_splits.json")

CALIBRATION_PATH = Path("artifacts/evaluation/production_thresholds.json")

OUTPUT_PATH = Path("artifacts/evaluation/day15_optimized_production_validation.json")

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

    model = mobilenet_v3_small(weights=weights)

    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================


def preprocess_image(image_path):

    image_path = resolve_path(image_path)

    image = Image.open(image_path).convert("RGB")

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

    tensor = (tensor - MEAN) / STD

    return tensor


# ============================================================
# FEATURE EXTRACTION
# ============================================================


@torch.no_grad()
def extract_layers(
    model,
    image_tensor,
):

    x = image_tensor

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

        image = preprocess_image(path)

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
# REFERENCE SCORING
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

        reference_chunk = reference_bank[start:end]

        similarity = query @ reference_chunk.T

        chunk_best = similarity.max(dim=1).values

        best_similarity = torch.maximum(
            best_similarity,
            chunk_best,
        )

    squared_distance = 2.0 - 2.0 * best_similarity

    squared_distance = torch.clamp(
        squared_distance,
        min=0.0,
    )

    distances = torch.sqrt(squared_distance)

    topk = torch.topk(
        distances,
        k=TOP_K,
        largest=True,
    ).values

    return topk.mean()


# ============================================================
# REFERENCE SCORING — ORIGINAL VALIDATED FORM
# ============================================================


@torch.no_grad()
def score_cdist_reference(
    query,
    reference_bank,
):

    distances = torch.cdist(
        query,
        reference_bank,
        p=2,
    )

    nearest_distances = distances.min(dim=1).values

    topk = torch.topk(
        nearest_distances,
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
    category_data = calibration["categories"][category]

    # Production threshold artifact stores
    # the selected P99 threshold directly.
    if "threshold" in category_data:
        return float(category_data["threshold"])

    # Also support an explicit P99 field
    # if present in the artifact.
    if "p99_threshold" in category_data:
        return float(category_data["p99_threshold"])

    # Fallback for artifacts containing
    # percentile candidates.
    if "percentile_candidates" in category_data:

        for candidate in category_data["percentile_candidates"]:

            if float(candidate["percentile"]) == 99.0:

                return float(candidate["threshold"])

    raise ValueError(
        f"Could not find a production " f"threshold for category: {category}"
    )


def classify(
    score,
    threshold,
):

    review_threshold = threshold * REVIEW_MULTIPLIER

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
    print("VISYN — DAY 15")
    print("OPTIMIZED PRODUCTION PATH VALIDATION")
    print("=" * 72)

    print()
    print("Architecture:")
    print("  MobileNetV3-Small")
    print("  L4 + L8")
    print("  Raw 50/50 fusion")
    print("  Euclidean-equivalent normalized scoring")
    print("  TOP5")
    print(f"  Chunk size = {CHUNK_SIZE}")

    split = load_json(NORMAL_SPLIT_PATH)

    calibration = load_json(CALIBRATION_PATH)

    model = load_model()

    results = {
        "architecture": "MobileNetV3-Small L4+L8",
        "fusion": "raw_50_50",
        "distance": "Euclidean-equivalent",
        "aggregation": "TOP5",
        "chunk_size": CHUNK_SIZE,
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
        print(f"CATEGORY: {category}")
        print("=" * 72)

        reference_paths = split["categories"][category]["reference"]

        development_paths = split["categories"][category]["development"]

        test_paths = development_paths[:5]

        print(f"Reference images: " f"{len(reference_paths)}")

        print(f"Validation images: " f"{len(test_paths)}")

        print()
        print("Building reference banks...")

        l4_bank, l8_bank = build_reference_bank(
            model,
            reference_paths,
        )

        threshold = get_p99_threshold(
            calibration,
            category,
        )

        category_results = {
            "threshold": threshold,
            "samples": [],
        }

        for path in test_paths:

            image = preprocess_image(path)

            l4_query, l8_query = extract_layers(
                model,
                image,
            )

            # ------------------------------------------------
            # Original validated scorer
            # ------------------------------------------------

            l4_original = score_cdist_reference(
                l4_query,
                l4_bank,
            )

            l8_original = score_cdist_reference(
                l8_query,
                l8_bank,
            )

            # ------------------------------------------------
            # Optimized production scorer
            # ------------------------------------------------

            l4_optimized = score_chunked(
                l4_query,
                l4_bank,
            )

            l8_optimized = score_chunked(
                l8_query,
                l8_bank,
            )

            # ------------------------------------------------
            # Differences
            # ------------------------------------------------

            l4_difference = abs((l4_original - l4_optimized).item())

            l8_difference = abs((l8_original - l8_optimized).item())

            original_fusion = (
                FUSION_WEIGHT_L4 * l4_original + FUSION_WEIGHT_L8 * l8_original
            )

            optimized_fusion = (
                FUSION_WEIGHT_L4 * l4_optimized + FUSION_WEIGHT_L8 * l8_optimized
            )

            fusion_difference = abs((original_fusion - optimized_fusion).item())

            # ------------------------------------------------
            # Decision equivalence
            # ------------------------------------------------

            original_decision, original_review = classify(
                original_fusion.item(),
                threshold,
            )

            optimized_decision, optimized_review = classify(
                optimized_fusion.item(),
                threshold,
            )

            decision_match = original_decision == optimized_decision

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
            print(f"{path}")

            print(f"  L4 difference: " f"{l4_difference:.10f}")

            print(f"  L8 difference: " f"{l8_difference:.10f}")

            print(f"  Fusion difference: " f"{fusion_difference:.10f}")

            print(f"  Decision: " f"{optimized_decision}")

            print(f"  Decision match: " f"{decision_match}")

            category_results["samples"].append(
                {
                    "path": str(path),
                    "l4_original": float(l4_original.item()),
                    "l4_optimized": float(l4_optimized.item()),
                    "l4_difference": float(l4_difference),
                    "l8_original": float(l8_original.item()),
                    "l8_optimized": float(l8_optimized.item()),
                    "l8_difference": float(l8_difference),
                    "fusion_original": float(original_fusion.item()),
                    "fusion_optimized": float(optimized_fusion.item()),
                    "fusion_difference": float(fusion_difference),
                    "threshold": float(threshold),
                    "original_review_threshold": float(original_review),
                    "optimized_review_threshold": float(optimized_review),
                    "original_decision": original_decision,
                    "optimized_decision": optimized_decision,
                    "decision_match": decision_match,
                }
            )

        results["categories"][category] = category_results

    equivalence_pass = (
        global_max_l4_difference < 1e-5
        and global_max_l8_difference < 1e-5
        and global_max_fusion_difference < 1e-5
        and decision_matches == decision_total
    )

    results["global_max_l4_difference"] = float(global_max_l4_difference)

    results["global_max_l8_difference"] = float(global_max_l8_difference)

    results["global_max_fusion_difference"] = float(global_max_fusion_difference)

    results["decision_matches"] = int(decision_matches)

    results["decision_total"] = int(decision_total)

    results["equivalence_pass"] = bool(equivalence_pass)

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

    print("L4 numerical equivalence: " f"{global_max_l4_difference:.10f}")

    print("L8 numerical equivalence: " f"{global_max_l8_difference:.10f}")

    print("Fusion numerical equivalence: " f"{global_max_fusion_difference:.10f}")

    print(f"Decision matches: " f"{decision_matches}/{decision_total}")

    if equivalence_pass:
        print()
        print("OPTIMIZED PRODUCTION PATH: PASS")
    else:
        print()
        print("OPTIMIZED PRODUCTION PATH: FAIL")

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
