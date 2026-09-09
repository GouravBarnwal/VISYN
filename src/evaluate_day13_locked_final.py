import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ============================================================
# VISYN — DAY 14
# CORRECTED LOCKED FINAL-SET EVALUATION
# ============================================================

NORMAL_SPLIT_PATH = Path("artifacts/splits/normal_splits.json")

DEFECT_SPLIT_PATH = Path("artifacts/splits/defect_splits.json")

OUTPUT_PATH = Path("artifacts/evaluation/day14/day14_locked_final_results.json")

DATA_ROOT = Path("data")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

TOP_K = 5

# Canonical VISYN preprocessing
IMAGE_SIZE = 224

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# DATA
# ============================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(relative_path):
    path = Path(relative_path)

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
# CANONICAL PREPROCESSING
# ============================================================


def preprocess_image(image_path):
    image_path = resolve_path(image_path)

    image = Image.open(image_path).convert("RGB")

    # IMPORTANT:
    # Direct resize — NO center crop.
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

    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    return tensor


# ============================================================
# FEATURE EXTRACTION
# ============================================================


@torch.no_grad()
def extract_layers(model, image_path):
    x = preprocess_image(image_path)

    layer4 = None
    layer8 = None

    for index, layer in enumerate(model.features):
        x = layer(x)

        if index == 4:
            layer4 = x.clone()

        if index == 8:
            layer8 = x.clone()

    if layer4 is None:
        raise RuntimeError("L4 feature was not captured.")

    if layer8 is None:
        raise RuntimeError("L8 feature was not captured.")

    # Convert:
    # [1, C, H, W]
    #
    # into:
    # [H*W, C]
    l4 = layer4.squeeze(0)
    l4 = l4.permute(1, 2, 0)
    l4 = l4.reshape(-1, l4.shape[-1])

    l8 = layer8.squeeze(0)
    l8 = l8.permute(1, 2, 0)
    l8 = l8.reshape(-1, l8.shape[-1])

    # Per-patch L2 normalization.
    l4 = F.normalize(l4, p=2, dim=1)
    l8 = F.normalize(l8, p=2, dim=1)

    return l4, l8


# ============================================================
# REFERENCE BANK
# ============================================================


def build_reference_bank(model, paths):
    l4_features = []
    l8_features = []

    for path in paths:
        l4, l8 = extract_layers(model, path)

        l4_features.append(l4)
        l8_features.append(l8)

    l4_bank = torch.cat(l4_features, dim=0)
    l8_bank = torch.cat(l8_features, dim=0)

    return l4_bank, l8_bank


# ============================================================
# IMAGE SCORE
# ============================================================


@torch.no_grad()
def score_against_bank(query_features, reference_bank):
    distances = torch.cdist(
        query_features,
        reference_bank,
        p=2,
    )

    # For every query patch, find its nearest reference patch.
    nearest = distances.min(dim=1).values

    if nearest.numel() < TOP_K:
        raise RuntimeError(
            f"Query contains only {nearest.numel()} patches; "
            f"TOP5 aggregation is impossible."
        )

    topk = torch.topk(
        nearest,
        k=TOP_K,
        largest=True,
    ).values

    return float(topk.mean().item())


def score_image(model, image_path, l4_bank, l8_bank):
    l4, l8 = extract_layers(model, image_path)

    l4_score = score_against_bank(
        l4,
        l4_bank,
    )

    l8_score = score_against_bank(
        l8,
        l8_bank,
    )

    # Selected Day 14 architecture:
    # raw 50/50 L4 + L8 fusion.
    fusion_score = 0.5 * l4_score + 0.5 * l8_score

    return {
        "l4": l4_score,
        "l8": l8_score,
        "fusion": fusion_score,
    }


# ============================================================
# DATASET SCORING
# ============================================================


def score_final_defects(
    model,
    paths,
    l4_bank,
    l8_bank,
):
    scores = []

    for path in paths:
        result = score_image(
            model,
            path,
            l4_bank,
            l8_bank,
        )

        scores.append(result)

    return scores


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    normal_scores,
    defect_scores,
):
    y_true = np.concatenate(
        [
            np.zeros(len(normal_scores)),
            np.ones(len(defect_scores)),
        ]
    )

    y_score = np.concatenate(
        [
            np.asarray(normal_scores),
            np.asarray(defect_scores),
        ]
    )

    auroc = roc_auc_score(
        y_true,
        y_score,
    )

    ap = average_precision_score(
        y_true,
        y_score,
    )

    return {
        "auroc": float(auroc),
        "average_precision": float(ap),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 72)
    print("VISYN — DAY 14")
    print("CORRECTED LOCKED FINAL-SET EVALUATION")
    print("=" * 72)

    print()
    print("Protocol:")
    print("  Preprocessing: direct 224 x 224 resize")
    print("  Scaling: /255")
    print("  Normalization: ImageNet mean/std")
    print("  Feature layers: L4 + L8")
    print("  Distance: Euclidean (torch.cdist)")
    print("  Aggregation: TOP5")
    print("  Fusion: raw 50/50 L4 + L8")
    print("  Final defect set: LOCKED")
    print()

    normal_split = load_json(NORMAL_SPLIT_PATH)

    defect_split = load_json(DEFECT_SPLIT_PATH)

    model = load_model()

    results = {
        "experiment": ("VISYN Day 14 corrected " "locked final-set evaluation"),
        "final_test_used": True,
        "final_test_used_for_tuning": False,
        "preprocessing": {
            "resize": "direct_224x224",
            "scaling": "/255",
            "normalization": "ImageNet mean/std",
            "center_crop": False,
        },
        "model": "MobileNetV3-Small",
        "layers": {
            "L4": {
                "index": 4,
                "channels": 40,
                "spatial": "14x14",
            },
            "L8": {
                "index": 8,
                "channels": 48,
                "spatial": "14x14",
            },
        },
        "distance": "Euclidean",
        "aggregation": "TOP5",
        "fusion": {
            "type": "raw_weighted_average",
            "L4_weight": 0.5,
            "L8_weight": 0.5,
        },
        "categories": {},
    }

    macro_l4 = []
    macro_l8 = []
    macro_fusion = []

    macro_ap_l4 = []
    macro_ap_l8 = []
    macro_ap_fusion = []

    for category in CATEGORIES:
        print("=" * 72)
        print(f"CATEGORY: {category}")
        print("=" * 72)

        reference_paths = normal_split["categories"][category]["reference"]

        normal_dev_paths = normal_split["categories"][category]["development"]

        final_defect_paths = defect_split["categories"][category]["final_test"]

        print(f"Reference images: {len(reference_paths)}")

        print(f"Normal development: " f"{len(normal_dev_paths)}")

        print(f"Locked final defects: " f"{len(final_defect_paths)}")

        print()
        print("Building reference banks...")

        l4_bank, l8_bank = build_reference_bank(
            model,
            reference_paths,
        )

        print(f"L4 bank shape: " f"{tuple(l4_bank.shape)}")

        print(f"L8 bank shape: " f"{tuple(l8_bank.shape)}")

        print()
        print("Scoring normal development...")

        normal_scores = score_final_defects(
            model,
            normal_dev_paths,
            l4_bank,
            l8_bank,
        )

        print("Scoring locked final defects...")

        final_scores = score_final_defects(
            model,
            final_defect_paths,
            l4_bank,
            l8_bank,
        )

        normal_l4 = [item["l4"] for item in normal_scores]

        normal_l8 = [item["l8"] for item in normal_scores]

        normal_fusion = [item["fusion"] for item in normal_scores]

        defect_l4 = [item["l4"] for item in final_scores]

        defect_l8 = [item["l8"] for item in final_scores]

        defect_fusion = [item["fusion"] for item in final_scores]

        l4_metrics = calculate_metrics(
            normal_l4,
            defect_l4,
        )

        l8_metrics = calculate_metrics(
            normal_l8,
            defect_l8,
        )

        fusion_metrics = calculate_metrics(
            normal_fusion,
            defect_fusion,
        )

        print()
        print(
            f"L4     AUROC="
            f"{l4_metrics['auroc']:.4f} "
            f"AP="
            f"{l4_metrics['average_precision']:.4f}"
        )

        print(
            f"L8     AUROC="
            f"{l8_metrics['auroc']:.4f} "
            f"AP="
            f"{l8_metrics['average_precision']:.4f}"
        )

        print(
            f"Fusion  AUROC="
            f"{fusion_metrics['auroc']:.4f} "
            f"AP="
            f"{fusion_metrics['average_precision']:.4f}"
        )

        results["categories"][category] = {
            "reference_count": len(reference_paths),
            "normal_development_count": len(normal_dev_paths),
            "final_defect_count": len(final_defect_paths),
            "l4": {
                "auroc": l4_metrics["auroc"],
                "average_precision": l4_metrics["average_precision"],
            },
            "l8": {
                "auroc": l8_metrics["auroc"],
                "average_precision": l8_metrics["average_precision"],
            },
            "fusion_50_50": {
                "auroc": fusion_metrics["auroc"],
                "average_precision": fusion_metrics["average_precision"],
            },
        }

        macro_l4.append(l4_metrics["auroc"])
        macro_l8.append(l8_metrics["auroc"])
        macro_fusion.append(fusion_metrics["auroc"])

        macro_ap_l4.append(l4_metrics["average_precision"])
        macro_ap_l8.append(l8_metrics["average_precision"])
        macro_ap_fusion.append(fusion_metrics["average_precision"])

    results["macro_average"] = {
        "L4": {
            "auroc": float(np.mean(macro_l4)),
            "average_precision": float(np.mean(macro_ap_l4)),
        },
        "L8": {
            "auroc": float(np.mean(macro_l8)),
            "average_precision": float(np.mean(macro_ap_l8)),
        },
        "fusion_50_50": {
            "auroc": float(np.mean(macro_fusion)),
            "average_precision": float(np.mean(macro_ap_fusion)),
        },
    }

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
    print("MACRO LOCKED-FINAL RESULTS")
    print("=" * 72)

    print(
        f"L4      AUROC="
        f"{results['macro_average']['L4']['auroc']:.4f} "
        f"AP="
        f"{results['macro_average']['L4']['average_precision']:.4f}"
    )

    print(
        f"L8      AUROC="
        f"{results['macro_average']['L8']['auroc']:.4f} "
        f"AP="
        f"{results['macro_average']['L8']['average_precision']:.4f}"
    )

    print(
        f"Fusion  AUROC="
        f"{results['macro_average']['fusion_50_50']['auroc']:.4f} "
        f"AP="
        f"{results['macro_average']['fusion_50_50']['average_precision']:.4f}"
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("FINAL SET WAS USED ONLY FOR EVALUATION — " "NOT FOR TUNING.")


if __name__ == "__main__":
    main()
