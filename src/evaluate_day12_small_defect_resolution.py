import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("data")

SPLIT_PATH = Path(
    "artifacts/splits/normal_splits.json"
)

DEFECT_SPLIT_PATH = Path(
    "artifacts/splits/defect_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day12_small_defect_resolution.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")

AREA_BINS = [
    ("tiny", 0.0, 0.10),
    ("small", 0.10, 0.50),
    ("medium", 0.50, 2.00),
    ("large", 2.00, 100.01),
]


# ============================================================
# Image preprocessing
# ============================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def load_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        image
    ).permute(2, 0, 1)

    tensor = (
        tensor - MEAN
    ) / STD

    return tensor


# ============================================================
# MobileNet
# ============================================================

def build_model():

    from torchvision.models import (
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights,
    )

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    model.eval()
    model.to(DEVICE)

    return model


class FeatureExtractor(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.features = model.features

    @torch.no_grad()
    def forward(self, x):

        outputs = {}

        # L3:
        # features[:4] gives approximately
        # 24 channels × 28 × 28
        l3 = x

        for index, layer in enumerate(
            self.features
        ):

            l3 = layer(l3)

            if index == 3:
                outputs["L3"] = l3

            if index == 8:
                outputs["L8"] = l3
                break

        return outputs


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(
    extractor,
    image_path,
):

    image = load_image(
        image_path
    )

    image = image.unsqueeze(0).to(
        DEVICE
    )

    outputs = extractor(image)

    result = {}

    for layer_name, feature in outputs.items():

        feature = feature[0]

        # C × H × W -> HW × C

        patches = feature.permute(
            1, 2, 0
        ).reshape(
            -1,
            feature.shape[0],
        )

        patches = torch.nn.functional.normalize(
            patches,
            dim=1,
        )

        result[layer_name] = patches

    return result


# ============================================================
# Reference banks
# ============================================================

def build_reference_banks(
    extractor,
    split,
):

    banks = {}

    for category in CATEGORIES:

        print(
            f"Building reference bank: "
            f"{category}"
        )

        category_bank = {
            "L3": [],
            "L8": [],
        }

        paths = split[
            "categories"
        ][category][
            "reference"
        ]

        for relative_path in paths:

            path = DATA_ROOT / relative_path

            features = extract_features(
                extractor,
                path,
            )

            category_bank["L3"].append(
                features["L3"]
            )

            category_bank["L8"].append(
                features["L8"]
            )

        banks[category] = {}

        for layer_name in [
            "L3",
            "L8",
        ]:

            banks[category][
                layer_name
            ] = torch.cat(
                category_bank[layer_name],
                dim=0,
            )

        print(
            f"  L3 bank: "
            f"{banks[category]['L3'].shape}"
        )

        print(
            f"  L8 bank: "
            f"{banks[category]['L8'].shape}"
        )

    return banks


# ============================================================
# Anomaly score
# ============================================================

@torch.no_grad()
def score_image(
    features,
    reference_bank,
):

    scores = {}

    for layer_name in [
        "L3",
        "L8",
    ]:

        query = features[layer_name]

        reference = reference_bank[
            layer_name
        ]

        distances = torch.cdist(
            query,
            reference,
            p=2,
        )

        nearest = distances.min(
            dim=1
        ).values

        # TOP5 aggregation

        k = min(
            5,
            nearest.numel(),
        )

        top_values = torch.topk(
            nearest,
            k=k,
            largest=True,
        ).values

        scores[layer_name] = float(
            top_values.mean()
        )

    return scores


# ============================================================
# Mask
# ============================================================

def find_mask(image_path):

    image_path = Path(
        image_path
    )

    parts = image_path.parts

    if len(parts) < 5:
        return None

    category = parts[1]
    defect_type = parts[-2]
    stem = image_path.stem

    mask_path = (
        DATA_ROOT
        / category
        / "ground_truth"
        / defect_type
        / f"{stem}_mask.png"
    )

    if mask_path.exists():
        return mask_path

    return None


def mask_area_percent(
    image_path,
):

    mask_path = find_mask(
        image_path
    )

    if mask_path is None:
        return None

    mask = np.asarray(
        Image.open(mask_path)
    )

    if mask.ndim == 3:
        mask = mask[..., 0]

    positive = np.count_nonzero(
        mask > 0
    )

    total = mask.shape[0] * mask.shape[1]

    if total == 0:
        return None

    return (
        positive / total * 100.0
    )


def assign_area_bin(area):

    for name, lower, upper in AREA_BINS:

        if (
            area >= lower
            and area < upper
        ):
            return name

    return None


# ============================================================
# Development defects
# ============================================================

def load_defect_development():

    data = json.load(
        open(
            DEFECT_SPLIT_PATH,
            "r",
            encoding="utf-8",
        )
    )

    result = {}

    for category in CATEGORIES:

        result[category] = data[
            "categories"
        ][category][
            "development"
        ]

    return result


# ============================================================
# Normal development samples
# ============================================================

def load_normal_development():

    data = json.load(
        open(
            SPLIT_PATH,
            "r",
            encoding="utf-8",
        )
    )

    result = {}

    for category in CATEGORIES:

        result[category] = data[
            "categories"
        ][category][
            "development"
        ]

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "DAY 12 — SMALL-DEFECT RESOLUTION TEST"
    )
    print("=" * 78)

    print(
        "\nDevelopment split only."
    )

    print(
        "Final test is NOT used."
    )

    print(
        "\nBuilding MobileNetV3-Small..."
    )

    model = build_model()

    extractor = FeatureExtractor(
        model
    )

    normal_split = json.load(
        open(
            SPLIT_PATH,
            "r",
            encoding="utf-8",
        )
    )

    reference_banks = build_reference_banks(
        extractor,
        normal_split,
    )

    normal_paths = (
        load_normal_development()
    )

    defect_paths = (
        load_defect_development()
    )

    results = {
        "split": "development",
        "final_test_used": False,
        "model": "MobileNetV3-Small",
        "aggregation": "TOP5",
        "layers": [
            "L3",
            "L8",
        ],
        "area_bins": [
            {
                "name": name,
                "lower_percent": lower,
                "upper_percent": upper,
            }
            for name, lower, upper
            in AREA_BINS
        ],
        "categories": {},
    }

    # --------------------------------------------------------
    # Evaluate every category
    # --------------------------------------------------------

    for category in CATEGORIES:

        print("\n" + "=" * 78)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 78)

        normal_records = []
        defect_records = []

        # ----------------------------------------------------
        # Normal development
        # ----------------------------------------------------

        for relative_path in normal_paths[
            category
        ]:

            path = (
                DATA_ROOT
                / relative_path
            )

            features = extract_features(
                extractor,
                path,
            )

            scores = score_image(
                features,
                reference_banks[
                    category
                ],
            )

            normal_records.append(
                {
                    "path": str(path),
                    "label": 0,
                    "L3": scores["L3"],
                    "L8": scores["L8"],
                }
            )

        # ----------------------------------------------------
        # Defect development
        # ----------------------------------------------------

        missing_masks = 0

        for relative_path in defect_paths[
            category
        ]:

            path = Path(
                relative_path
            )

            if not path.is_absolute():

                if (
                    path.parts
                    and path.parts[0]
                    == "data"
                ):
                    path = path
                else:
                    path = (
                        DATA_ROOT / path
                    )

            area = mask_area_percent(
                path
            )

            if area is None:

                missing_masks += 1

                continue

            features = extract_features(
                extractor,
                path,
            )

            scores = score_image(
                features,
                reference_banks[
                    category
                ],
            )

            defect_records.append(
                {
                    "path": str(path),
                    "label": 1,
                    "area_percent": float(
                        area
                    ),
                    "area_bin": assign_area_bin(
                        area
                    ),
                    "L3": scores["L3"],
                    "L8": scores["L8"],
                }
            )

        print(
            f"Normal samples: "
            f"{len(normal_records)}"
        )

        print(
            f"Defect samples: "
            f"{len(defect_records)}"
        )

        print(
            f"Missing masks: "
            f"{missing_masks}"
        )

        category_result = {
            "normal_count": len(
                normal_records
            ),
            "defect_count": len(
                defect_records
            ),
            "bins": {},
        }

        # ----------------------------------------------------
        # Size-specific detection
        # ----------------------------------------------------

        for (
            bin_name,
            _,
            _,
        ) in AREA_BINS:

            bin_defects = [
                record
                for record in defect_records
                if record["area_bin"]
                == bin_name
            ]

            print(
                f"\n{bin_name.upper()}"
            )

            print(
                f"Defects: "
                f"{len(bin_defects)}"
            )

            bin_result = {
                "defect_count": len(
                    bin_defects
                )
            }

            if not bin_defects:

                category_result[
                    "bins"
                ][bin_name] = bin_result

                continue

            labels = (
                [0] * len(normal_records)
                + [1] * len(bin_defects)
            )

            for layer_name in [
                "L3",
                "L8",
            ]:

                scores = (
                    [
                        record[layer_name]
                        for record
                        in normal_records
                    ]
                    + [
                        record[layer_name]
                        for record
                        in bin_defects
                    ]
                )

                scores = np.asarray(
                    scores,
                    dtype=np.float64,
                )

                labels_array = np.asarray(
                    labels,
                    dtype=np.int64,
                )

                auroc = roc_auc_score(
                    labels_array,
                    scores,
                )

                ap = average_precision_score(
                    labels_array,
                    scores,
                )

                bin_result[
                    f"{layer_name}_AUROC"
                ] = float(auroc)

                bin_result[
                    f"{layer_name}_AP"
                ] = float(ap)

                print(
                    f"  {layer_name}: "
                    f"AUROC={auroc:.4f} "
                    f"AP={ap:.4f}"
                )

            category_result[
                "bins"
            ][bin_name] = bin_result

        results[
            "categories"
        ][category] = category_result

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    print("\n" + "=" * 78)
    print(
        "RESULTS SAVED:"
    )
    print(OUTPUT_PATH)
    print("=" * 78)

    print(
        "\nDAY 12 EXPERIMENT 7 COMPLETE"
    )


if __name__ == "__main__":
    main()