import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import mobilenet_v3_small


# ============================================================
# VISIONFORGE — DAY 11
# L4 + L8 FUSION DEFECT-TYPE ANALYSIS
# ============================================================

DATA_ROOT = Path("data")

SPLITS_PATH = Path(
    "artifacts/splits/defect_splits.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day11_l4_l8_defect_types.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

L4_WEIGHT = 0.5
L8_WEIGHT = 0.5

TOP_K = 5

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

def load_model():
    model = mobilenet_v3_small(
        weights="DEFAULT"
    )

    model.eval()
    model.to(DEVICE)

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))

    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    array = (array - mean) / std

    array = np.transpose(
        array,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    model,
    image_path,
):
    x = preprocess_image(image_path)

    features = x

    l4 = None
    l8 = None

    for index, layer in enumerate(
        model.features
    ):
        features = layer(features)

        if index == 4:
            l4 = features.clone()

        if index == 8:
            l8 = features.clone()

    return (
        l4.squeeze(0),
        l8.squeeze(0),
    )


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(path_string):
    path = Path(path_string)

    if path.is_absolute():
        return path

    return DATA_ROOT / path


# ============================================================
# DEFECT TYPE
# ============================================================

def get_defect_type(path_string):
    path = Path(path_string)

    return path.parent.name


# ============================================================
# REFERENCE BANK
# ============================================================

def build_reference_bank(
    model,
    paths,
    layer_name,
):
    descriptors = []

    total = len(paths)

    for index, path_string in enumerate(
        paths,
        start=1,
    ):
        path = resolve_path(
            path_string
        )

        l4, l8 = extract_features(
            model,
            path,
        )

        feature_map = (
            l4
            if layer_name == "L4"
            else l8
        )

        channels, height, width = (
            feature_map.shape
        )

        patches = (
            feature_map
            .permute(1, 2, 0)
            .reshape(-1, channels)
        )

        descriptors.append(
            patches.cpu()
        )

        if (
            index % 50 == 0
            or index == total
        ):
            print(
                f"{layer_name} reference: "
                f"{index}/{total}"
            )

    return torch.cat(
        descriptors,
        dim=0,
    )


# ============================================================
# IMAGE SCORE
# ============================================================

@torch.no_grad()
def score_image(
    model,
    image_path,
    l4_bank,
    l8_bank,
):
    l4, l8 = extract_features(
        model,
        image_path,
    )

    def score_layer(
        feature_map,
        bank,
    ):
        channels, height, width = (
            feature_map.shape
        )

        patches = (
            feature_map
            .permute(1, 2, 0)
            .reshape(-1, channels)
        )

        distances = torch.cdist(
            patches,
            bank,
            p=2,
        )

        nearest = distances.min(
            dim=1
        ).values

        top_k = min(
            TOP_K,
            nearest.numel(),
        )

        top_values = torch.topk(
            nearest,
            k=top_k,
            largest=True,
        ).values

        return float(
            top_values.mean().item()
        )

    l4_score = score_layer(
        l4,
        l4_bank,
    )

    l8_score = score_layer(
        l8,
        l8_bank,
    )

    return (
        l4_score,
        l8_score,
    )


# ============================================================
# ROBUST NORMALIZATION
# ============================================================

def fit_normalization(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    return {
        "median": median,
        "mad": mad,
    }


def robust_normalize(
    values,
    median,
    mad,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    scale = max(
        1.4826 * mad,
        1e-8,
    )

    return (
        values - median
    ) / scale


# ============================================================
# LOAD SPLITS
# ============================================================

def load_splits():
    with open(
        SPLITS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = load_splits()

    model = load_model()

    results = {}

    all_fusion_aurocs = []
    all_fusion_aps = []

    print()
    print("=" * 72)
    print(
        "VISIONFORGE — DAY 11 "
        "L4 + L8 DEFECT-TYPE ANALYSIS"
    )
    print("=" * 72)
    print()
    print(
        "L4 weight:",
        L4_WEIGHT,
    )
    print(
        "L8 weight:",
        L8_WEIGHT,
    )
    print(
        "Distance: Euclidean (torch.cdist)"
    )
    print(
        "Aggregation: TOP5"
    )
    print(
        "Normalization: median/MAD"
    )
    print(
        "Normalization fitted "
        "on normal development only"
    )
    print(
        "Split: canonical development"
    )
    print(
        "Final locked defect set: NOT USED"
    )
    print()

    for category in CATEGORIES:

        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        category_split = (
            splits["categories"][category]
        )

        # ----------------------------------------------------
        # DEFECT DEVELOPMENT
        # ----------------------------------------------------

        defect_dev_paths = (
            category_split["development"]
        )

        # ----------------------------------------------------
        # NORMAL DEVELOPMENT
        #
        # Normal development images come from
        # the canonical normal split.
        # ----------------------------------------------------

        normal_split_path = Path(
            "artifacts/splits/"
            "normal_splits.json"
        )

        with open(
            normal_split_path,
            "r",
            encoding="utf-8",
        ) as f:
            normal_splits = json.load(f)

        normal_category = (
            normal_splits[
                "categories"
            ][category]
        )

        reference_paths = (
            normal_category["reference"]
        )

        normal_dev_paths = (
            normal_category["development"]
        )

        # ----------------------------------------------------
        # L4 BANK
        # ----------------------------------------------------

        print()
        print(
            "Building L4 reference bank..."
        )

        l4_bank = build_reference_bank(
            model,
            reference_paths,
            "L4",
        )

        print(
            "L4 bank:",
            tuple(l4_bank.shape),
        )

        # ----------------------------------------------------
        # L8 BANK
        # ----------------------------------------------------

        print()
        print(
            "Building L8 reference bank..."
        )

        l8_bank = build_reference_bank(
            model,
            reference_paths,
            "L8",
        )

        print(
            "L8 bank:",
            tuple(l8_bank.shape),
        )

        # ----------------------------------------------------
        # NORMAL DEVELOPMENT SCORES
        # ----------------------------------------------------

        l4_normal = []
        l8_normal = []

        print()
        print(
            "Scoring normal development..."
        )

        for index, path_string in enumerate(
            normal_dev_paths,
            start=1,
        ):
            path = resolve_path(
                path_string
            )

            l4_score, l8_score = (
                score_image(
                    model,
                    path,
                    l4_bank,
                    l8_bank,
                )
            )

            l4_normal.append(
                l4_score
            )

            l8_normal.append(
                l8_score
            )

            if (
                index % 25 == 0
                or index
                == len(normal_dev_paths)
            ):
                print(
                    f"normal: "
                    f"{index}/"
                    f"{len(normal_dev_paths)}"
                )

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        l4_norm = fit_normalization(
            l4_normal
        )

        l8_norm = fit_normalization(
            l8_normal
        )

        l4_normalized = (
            robust_normalize(
                l4_normal,
                l4_norm["median"],
                l4_norm["mad"],
            )
        )

        l8_normalized = (
            robust_normalize(
                l8_normal,
                l8_norm["median"],
                l8_norm["mad"],
            )
        )

        fusion_normal = (
            L4_WEIGHT
            * l4_normalized
            + L8_WEIGHT
            * l8_normalized
        )

        # ----------------------------------------------------
        # DEFECT DEVELOPMENT SCORES
        # ----------------------------------------------------

        defect_records = []

        print()
        print(
            "Scoring defect development..."
        )

        for index, path_string in enumerate(
            defect_dev_paths,
            start=1,
        ):

            path = resolve_path(
                path_string
            )

            l4_score, l8_score = (
                score_image(
                    model,
                    path,
                    l4_bank,
                    l8_bank,
                )
            )

            defect_type = (
                get_defect_type(
                    path_string
                )
            )

            defect_records.append(
                {
                    "path": path_string,
                    "defect_type": defect_type,
                    "l4_score": l4_score,
                    "l8_score": l8_score,
                }
            )

            if (
                index % 25 == 0
                or index
                == len(defect_dev_paths)
            ):
                print(
                    f"defect: "
                    f"{index}/"
                    f"{len(defect_dev_paths)}"
                )

        # ----------------------------------------------------
        # DEFECT TYPE ANALYSIS
        # ----------------------------------------------------

        type_results = {}

        defect_types = sorted(
            set(
                record["defect_type"]
                for record
                in defect_records
            )
        )

        for defect_type in defect_types:

            records = [
                record
                for record
                in defect_records
                if record["defect_type"]
                == defect_type
            ]

            l4_defect = np.asarray(
                [
                    record["l4_score"]
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            l8_defect = np.asarray(
                [
                    record["l8_score"]
                    for record
                    in records
                ],
                dtype=np.float64,
            )

            l4_defect_normalized = (
                robust_normalize(
                    l4_defect,
                    l4_norm["median"],
                    l4_norm["mad"],
                )
            )

            l8_defect_normalized = (
                robust_normalize(
                    l8_defect,
                    l8_norm["median"],
                    l8_norm["mad"],
                )
            )

            fusion_defect = (
                L4_WEIGHT
                * l4_defect_normalized
                + L8_WEIGHT
                * l8_defect_normalized
            )

            y_true = np.concatenate(
                [
                    np.zeros(
                        len(fusion_normal),
                        dtype=np.int32,
                    ),
                    np.ones(
                        len(fusion_defect),
                        dtype=np.int32,
                    ),
                ]
            )

            y_l4 = np.concatenate(
                [
                    l4_normalized,
                    l4_defect_normalized,
                ]
            )

            y_l8 = np.concatenate(
                [
                    l8_normalized,
                    l8_defect_normalized,
                ]
            )

            y_fusion = np.concatenate(
                [
                    fusion_normal,
                    fusion_defect,
                ]
            )

            l4_auroc = roc_auc_score(
                y_true,
                y_l4,
            )

            l8_auroc = roc_auc_score(
                y_true,
                y_l8,
            )

            fusion_auroc = roc_auc_score(
                y_true,
                y_fusion,
            )

            l4_ap = average_precision_score(
                y_true,
                y_l4,
            )

            l8_ap = average_precision_score(
                y_true,
                y_l8,
            )

            fusion_ap = average_precision_score(
                y_true,
                y_fusion,
            )

            type_results[
                defect_type
            ] = {
                "count": len(records),
                "L4": {
                    "AUROC": float(
                        l4_auroc
                    ),
                    "AP": float(
                        l4_ap
                    ),
                },
                "L8": {
                    "AUROC": float(
                        l8_auroc
                    ),
                    "AP": float(
                        l8_ap
                    ),
                },
                "fusion_50_50": {
                    "AUROC": float(
                        fusion_auroc
                    ),
                    "AP": float(
                        fusion_ap
                    ),
                },
                "fusion_minus_L8": {
                    "AUROC": float(
                        fusion_auroc
                        - l8_auroc
                    ),
                    "AP": float(
                        fusion_ap
                        - l8_ap
                    ),
                },
            }

            all_fusion_aurocs.append(
                fusion_auroc
            )

            all_fusion_aps.append(
                fusion_ap
            )

            print(
                f"{defect_type:25s} "
                f"L8={l8_auroc:.4f} "
                f"Fusion={fusion_auroc:.4f} "
                f"Delta="
                f"{fusion_auroc - l8_auroc:+.4f}"
            )

        results[category] = {
            "reference_count": len(
                reference_paths
            ),
            "normal_dev_count": len(
                normal_dev_paths
            ),
            "defect_dev_count": len(
                defect_dev_paths
            ),
            "normalization": {
                "L4": l4_norm,
                "L8": l8_norm,
            },
            "defect_types": type_results,
        }

    # ========================================================
    # SUMMARY
    # ========================================================

    macro_auroc = float(
        np.mean(
            all_fusion_aurocs
        )
    )

    macro_ap = float(
        np.mean(
            all_fusion_aps
        )
    )

    results["summary"] = {
        "categories": len(
            CATEGORIES
        ),
        "defect_types": len(
            all_fusion_aurocs
        ),
        "macro_defect_type_AUROC":
            macro_auroc,
        "macro_defect_type_AP":
            macro_ap,
        "L4_weight": L4_WEIGHT,
        "L8_weight": L8_WEIGHT,
        "top_k": TOP_K,
        "seed": SEED,
    }

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
        "MACRO DEFECT-TYPE SUMMARY"
    )
    print("=" * 72)

    print(
        f"50/50 Fusion macro "
        f"AUROC: {macro_auroc:.4f}"
    )

    print(
        f"50/50 Fusion macro "
        f"AP: {macro_ap:.4f}"
    )

    print()
    print(
        "Saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()