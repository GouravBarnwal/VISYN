import json
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import MobileNet_V3_Small_Weights


DATA_ROOT = Path("data")
ARTIFACT_ROOT = Path("artifacts")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

NORMAL_SPLIT_PATH = (
    ARTIFACT_ROOT / "splits" / "normal_splits.json"
)

DEFECT_SPLIT_PATH = (
    ARTIFACT_ROOT / "splits" / "defect_splits.json"
)

OUTPUT_PATH = (
    ARTIFACT_ROOT
    / "evaluation"
    / "day14_normalization_ablation.json"
)

DEVICE = torch.device("cpu")

L4_INDEX = 4
L8_INDEX = 8
TOP_K = 5


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(relative_path):
    path = Path(relative_path)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


def load_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    model.eval()
    model.to(DEVICE)

    return model


def preprocess_image(image_path):
    """
    Canonical VisionForge preprocessing.

    This MUST remain identical to the established
    Day 11 production/evaluation protocol.
    """
    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    image = torch.from_numpy(
        image
    ).permute(
        2,
        0,
        1,
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    ).view(
        3,
        1,
        1,
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(
        3,
        1,
        1,
    )

    image = (
        image - mean
    ) / std

    return image.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def extract_layers(model, image_tensor):
    x = image_tensor
    outputs = {}

    for index, layer in enumerate(model.features):
        x = layer(x)

        if index in (L4_INDEX, L8_INDEX):
            outputs[index] = x.clone()

    return outputs


def spatial_features(feature_map):
    feature_map = feature_map.squeeze(0)

    return feature_map.permute(
        1,
        2,
        0,
    ).reshape(
        -1,
        feature_map.shape[0],
    )


def normalize_rows(features):
    return features / (
        torch.linalg.norm(
            features,
            dim=1,
            keepdim=True,
        )
        + 1e-12
    )


def build_reference_banks(
    model,
    reference_paths,
):
    l4_bank = []
    l8_bank = []

    for relative_path in reference_paths:
        image_path = resolve_image_path(
            relative_path
        )

        image_tensor = preprocess_image(
            image_path
        )

        outputs = extract_layers(
            model,
            image_tensor,
        )

        l4 = normalize_rows(
            spatial_features(
                outputs[L4_INDEX]
            )
        )

        l8 = normalize_rows(
            spatial_features(
                outputs[L8_INDEX]
            )
        )

        l4_bank.append(
            l4.cpu()
        )

        l8_bank.append(
            l8.cpu()
        )

    return {
        "L4": torch.cat(
            l4_bank,
            dim=0,
        ),
        "L8": torch.cat(
            l8_bank,
            dim=0,
        ),
    }


@torch.no_grad()
def top5_score(
    features,
    reference_bank,
):
    features = normalize_rows(features)

    distances = torch.cdist(
        features,
        reference_bank,
        p=2,
    )

    nearest = distances.min(
        dim=1
    ).values

    k = min(
        TOP_K,
        nearest.numel(),
    )

    return torch.topk(
        nearest,
        k=k,
        largest=True,
    ).values.mean().item()


def score_dataset(
    model,
    paths,
    reference_banks,
):
    l4_scores = []
    l8_scores = []

    for relative_path in paths:
        image_path = resolve_image_path(
            relative_path
        )

        image_tensor = preprocess_image(
            image_path
        )

        outputs = extract_layers(
            model,
            image_tensor,
        )

        l4_features = spatial_features(
            outputs[L4_INDEX]
        )

        l8_features = spatial_features(
            outputs[L8_INDEX]
        )

        l4_scores.append(
            top5_score(
                l4_features,
                reference_banks["L4"],
            )
        )

        l8_scores.append(
            top5_score(
                l8_features,
                reference_banks["L8"],
            )
        )

    return (
        np.asarray(
            l4_scores,
            dtype=np.float64,
        ),
        np.asarray(
            l8_scores,
            dtype=np.float64,
        ),
    )


def fit_normalization(scores):
    median = np.median(scores)

    mad = np.median(
        np.abs(
            scores - median
        )
    )

    if mad < 1e-12:
        mad = np.std(scores)

    if mad < 1e-12:
        mad = 1.0

    return {
        "median": float(median),
        "mad": float(mad),
    }


def normalize_scores(
    scores,
    statistics,
):
    return (
        scores
        - statistics["median"]
    ) / statistics["mad"]


def calculate_metrics(
    normal_scores,
    defect_scores,
):
    y_true = np.concatenate(
        [
            np.zeros(
                len(normal_scores),
                dtype=np.int32,
            ),
            np.ones(
                len(defect_scores),
                dtype=np.int32,
            ),
        ]
    )

    scores = np.concatenate(
        [
            normal_scores,
            defect_scores,
        ]
    )

    return {
        "auroc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
    }


def main():
    print("=" * 72)
    print("VISIONFORGE — DAY 14 NORMALIZATION ABLATION")
    print("=" * 72)

    print()
    print("Development-only experiment.")
    print("Final locked defect set: NOT USED")
    print()
    print(
        "Preprocessing: canonical Day 11 protocol"
    )
    print(
        "Resize: direct 224 × 224"
    )
    print(
        "Normalization: /255 + ImageNet mean/std"
    )
    print(
        "Distance: Euclidean (torch.cdist)"
    )
    print(
        "Aggregation: TOP5"
    )

    normal_split = load_json(
        NORMAL_SPLIT_PATH
    )

    defect_split = load_json(
        DEFECT_SPLIT_PATH
    )

    model = load_model()

    category_results = {}

    for category in CATEGORIES:
        print()
        print("=" * 72)
        print(f"CATEGORY: {category}")
        print("=" * 72)

        category_normal = (
            normal_split["categories"][category]
        )

        category_defect = (
            defect_split["categories"][category]
        )

        reference_paths = (
            category_normal["reference"]
        )

        normal_paths = (
            category_normal["development"]
        )

        defect_paths = (
            category_defect["development"]
        )

        print(
            f"Reference images: "
            f"{len(reference_paths)}"
        )

        print(
            f"Normal development: "
            f"{len(normal_paths)}"
        )

        print(
            f"Defect development: "
            f"{len(defect_paths)}"
        )

        print(
            "\nBuilding reference banks..."
        )

        reference_banks = build_reference_banks(
            model,
            reference_paths,
        )

        print(
            "Scoring normal development..."
        )

        normal_l4, normal_l8 = score_dataset(
            model,
            normal_paths,
            reference_banks,
        )

        print(
            "Scoring defect development..."
        )

        defect_l4, defect_l8 = score_dataset(
            model,
            defect_paths,
            reference_banks,
        )

        l4_statistics = fit_normalization(
            normal_l4
        )

        l8_statistics = fit_normalization(
            normal_l8
        )

        normal_l4_normalized = normalize_scores(
            normal_l4,
            l4_statistics,
        )

        defect_l4_normalized = normalize_scores(
            defect_l4,
            l4_statistics,
        )

        normal_l8_normalized = normalize_scores(
            normal_l8,
            l8_statistics,
        )

        defect_l8_normalized = normalize_scores(
            defect_l8,
            l8_statistics,
        )

        normal_raw_fusion = (
            0.5 * normal_l4
            + 0.5 * normal_l8
        )

        defect_raw_fusion = (
            0.5 * defect_l4
            + 0.5 * defect_l8
        )

        normal_normalized_fusion = (
            0.5 * normal_l4_normalized
            + 0.5 * normal_l8_normalized
        )

        defect_normalized_fusion = (
            0.5 * defect_l4_normalized
            + 0.5 * defect_l8_normalized
        )

        metrics = {
            "L4_raw": calculate_metrics(
                normal_l4,
                defect_l4,
            ),
            "L8_raw": calculate_metrics(
                normal_l8,
                defect_l8,
            ),
            "fusion_raw_50_50": calculate_metrics(
                normal_raw_fusion,
                defect_raw_fusion,
            ),
            "fusion_normalized_50_50": calculate_metrics(
                normal_normalized_fusion,
                defect_normalized_fusion,
            ),
        }

        category_results[category] = {
            "normal_count": len(normal_paths),
            "defect_count": len(defect_paths),
            "normalization": {
                "L4": l4_statistics,
                "L8": l8_statistics,
            },
            "metrics": metrics,
        }

        print()
        print(
            f"L4 raw: "
            f"AUROC={metrics['L4_raw']['auroc']:.4f} "
            f"AP={metrics['L4_raw']['average_precision']:.4f}"
        )

        print(
            f"L8 raw: "
            f"AUROC={metrics['L8_raw']['auroc']:.4f} "
            f"AP={metrics['L8_raw']['average_precision']:.4f}"
        )

        print(
            f"Raw 50/50 fusion: "
            f"AUROC={metrics['fusion_raw_50_50']['auroc']:.4f} "
            f"AP={metrics['fusion_raw_50_50']['average_precision']:.4f}"
        )

        print(
            f"Normalized 50/50 fusion: "
            f"AUROC={metrics['fusion_normalized_50_50']['auroc']:.4f} "
            f"AP={metrics['fusion_normalized_50_50']['average_precision']:.4f}"
        )

    methods = [
        "L4_raw",
        "L8_raw",
        "fusion_raw_50_50",
        "fusion_normalized_50_50",
    ]

    macro = {}

    for method in methods:
        macro[method] = {
            "auroc": float(
                np.mean(
                    [
                        category_results[category][
                            "metrics"
                        ][method]["auroc"]
                        for category in CATEGORIES
                    ]
                )
            ),
            "average_precision": float(
                np.mean(
                    [
                        category_results[category][
                            "metrics"
                        ][method][
                            "average_precision"
                        ]
                        for category in CATEGORIES
                    ]
                )
            ),
        }

    normalization_gain = (
        macro["fusion_normalized_50_50"]["auroc"]
        - macro["fusion_raw_50_50"]["auroc"]
    )

    print()
    print("=" * 72)
    print("MACRO NORMALIZATION ABLATION")
    print("=" * 72)

    for method, values in macro.items():
        print(
            f"{method:<30}"
            f"AUROC={values['auroc']:.4f} "
            f"AP={values['average_precision']:.4f}"
        )

    print()
    print(
        f"Normalization AUROC gain: "
        f"{normalization_gain:+.4f}"
    )

    output = {
        "experiment": (
            "Day 14 L4+L8 normalization ablation"
        ),
        "final_locked_set_used": False,
        "preprocessing": {
            "resize": "224x224_direct",
            "scale": "divide_by_255",
            "mean": [
                0.485,
                0.456,
                0.406,
            ],
            "std": [
                0.229,
                0.224,
                0.225,
            ],
        },
        "distance": "Euclidean",
        "aggregation": "TOP5",
        "normalization": (
            "median/MAD fitted on normal development only"
        ),
        "fusion_weights": {
            "L4": 0.5,
            "L8": 0.5,
        },
        "categories": category_results,
        "macro": macro,
        "normalized_fusion_gain_over_raw": float(
            normalization_gain
        ),
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
            output,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()