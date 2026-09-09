import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# CONFIG
# ============================================================

DEFECT_SPLITS_PATH = Path(
    "artifacts/splits/defect_splits.json"
)

BANK_DIR = Path(
    "artifacts/patch_banks/mobilenet_l8_spatial"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/localization"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mobilenet_l8_spatial_localization_dev.json"
)

DATA_ROOT = Path("data")

DEVICE = torch.device("cpu")

FEATURE_H = 14
FEATURE_W = 14
FEATURE_C = 48

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# PREPROCESSING
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = mobilenet_v3_small(
        weights=weights
    )

    feature_extractor = nn.Sequential(
        *list(model.features.children())[:9]
    )

    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    return feature_extractor


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(image_path):

    image_path = Path(image_path)

    if image_path.is_absolute():
        return image_path

    return DATA_ROOT / image_path


def find_mask(image_path):

    image_path = Path(image_path)

    category = image_path.parts[0]
    defect_type = image_path.parent.name
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


# ============================================================
# LOAD SPATIAL BANK
# ============================================================

def load_spatial_bank(category):

    bank_path = (
        BANK_DIR
        / f"{category}.npy"
    )

    if not bank_path.exists():

        raise FileNotFoundError(
            f"Spatial reference bank not found: "
            f"{bank_path}"
        )

    bank = np.load(
        bank_path
    ).astype(
        np.float32
    )

    expected_shape = (
        None,
        FEATURE_H,
        FEATURE_W,
        FEATURE_C,
    )

    if bank.ndim != 4:

        raise RuntimeError(
            f"Unexpected bank dimensions for "
            f"{category}: {bank.shape}"
        )

    if bank.shape[1:] != expected_shape[1:]:

        raise RuntimeError(
            f"Unexpected bank shape for "
            f"{category}: {bank.shape}"
        )

    # Defensive normalization.
    norms = np.linalg.norm(
        bank,
        axis=3,
        keepdims=True
    )

    bank = bank / np.maximum(
        norms,
        1e-12
    )

    return bank


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_map(
    model,
    image_path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = preprocess(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    feature_map = model(
        tensor
    ).squeeze(0)

    expected_shape = (
        FEATURE_C,
        FEATURE_H,
        FEATURE_W,
    )

    if tuple(feature_map.shape) != expected_shape:

        raise RuntimeError(
            f"Unexpected feature map shape: "
            f"{tuple(feature_map.shape)}"
        )

    return feature_map.cpu().numpy()


# ============================================================
# FEATURE MAP → SPATIAL DESCRIPTORS
# ============================================================

def feature_map_to_spatial_patches(
    feature_map
):

    # [48, 14, 14]
    #
    # → [14, 14, 48]

    patches = np.transpose(
        feature_map,
        (1, 2, 0)
    )

    norms = np.linalg.norm(
        patches,
        axis=2,
        keepdims=True
    )

    patches = patches / np.maximum(
        norms,
        1e-12
    )

    return patches.astype(
        np.float32
    )


# ============================================================
# SPATIAL PATCH ANOMALY SCORES
# ============================================================

def compute_spatial_patch_scores(
    patches,
    spatial_bank
):

    # Query:
    #     patches[r, c]
    #
    # Reference:
    #     spatial_bank[:, r, c]
    #
    # Therefore each query location is compared ONLY
    # with normal reference descriptors from the same
    # spatial location.

    heatmap = np.zeros(
        (
            FEATURE_H,
            FEATURE_W
        ),
        dtype=np.float32
    )

    for row in range(FEATURE_H):

        for col in range(FEATURE_W):

            query_descriptor = (
                patches[row, col]
            )

            reference_descriptors = (
                spatial_bank[:, row, col, :]
            )

            similarities = (
                reference_descriptors
                @ query_descriptor
            )

            best_similarity = np.max(
                similarities
            )

            heatmap[row, col] = (
                1.0 - best_similarity
            )

    return heatmap


# ============================================================
# UPSAMPLE HEATMAP
# ============================================================

def upsample_heatmap(
    low_res_heatmap,
    height,
    width
):

    heatmap = cv2.resize(
        low_res_heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    return heatmap.astype(
        np.float32
    )


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(
    mask_path,
    height,
    width
):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        raise RuntimeError(
            f"Could not read mask: "
            f"{mask_path}"
        )

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    return mask > 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — SPATIAL MOBILENET L8 LOCALIZATION"
    )
    print("=" * 72)

    print(
        "Representation: MobileNetV3-Small L8"
    )

    print(
        "Feature map: 48 x 14 x 14"
    )

    print(
        "Reference matching: SAME SPATIAL LOCATION"
    )

    print(
        "Evaluation: development defect split"
    )

    print(
        "Final test used: False"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = build_model()

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Load defect split
    # --------------------------------------------------------

    defect_split = json.loads(
        DEFECT_SPLITS_PATH.read_text(
            encoding="utf-8"
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(category.upper())
        print("-" * 72)

        spatial_bank = (
            load_spatial_bank(
                category
            )
        )

        print(
            f"Spatial bank: "
            f"{spatial_bank.shape}"
        )

        defect_entries = (
            defect_split[
                "categories"
            ][
                category
            ][
                "development"
            ]
        )

        print(
            f"Development defects: "
            f"{len(defect_entries)}"
        )

        category_scores = []
        category_labels = []

        valid_images = 0
        missing_masks = 0

        # ====================================================
        # IMAGE LOOP
        # ====================================================

        for image_path in defect_entries:

            image_path = Path(
                image_path
            )

            resolved_image_path = (
                resolve_image_path(
                    image_path
                )
            )

            # ------------------------------------------------
            # Mask
            # ------------------------------------------------

            mask_path = find_mask(
                image_path
            )

            if mask_path is None:

                print(
                    f"WARNING: mask not found: "
                    f"{image_path}"
                )

                missing_masks += 1

                continue

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

            if not resolved_image_path.exists():

                raise FileNotFoundError(
                    f"Image not found: "
                    f"{resolved_image_path}"
                )

            image = Image.open(
                resolved_image_path
            ).convert("RGB")

            width, height = image.size

            # ------------------------------------------------
            # Feature map
            # ------------------------------------------------

            feature_map = (
                extract_feature_map(
                    model,
                    resolved_image_path
                )
            )

            # ------------------------------------------------
            # Spatial descriptors
            # ------------------------------------------------

            patches = (
                feature_map_to_spatial_patches(
                    feature_map
                )
            )

            # ------------------------------------------------
            # Spatial anomaly map
            # ------------------------------------------------

            low_res_heatmap = (
                compute_spatial_patch_scores(
                    patches,
                    spatial_bank
                )
            )

            # ------------------------------------------------
            # Original-resolution heatmap
            # ------------------------------------------------

            heatmap = (
                upsample_heatmap(
                    low_res_heatmap,
                    height,
                    width
                )
            )

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            mask = load_mask(
                mask_path,
                height,
                width
            )

            # ------------------------------------------------
            # Accumulate pixel scores
            # ------------------------------------------------

            scores = heatmap.reshape(
                -1
            ).astype(
                np.float64
            )

            labels = mask.reshape(
                -1
            ).astype(
                np.uint8
            )

            if np.unique(labels).size < 2:

                continue

            category_scores.extend(
                scores.tolist()
            )

            category_labels.extend(
                labels.tolist()
            )

            valid_images += 1

        # ====================================================
        # CATEGORY PIXEL AUROC
        # ====================================================

        if (
            len(category_labels) > 0
            and len(
                np.unique(
                    category_labels
                )
            ) == 2
        ):

            category_auc = float(
                roc_auc_score(
                    np.asarray(
                        category_labels
                    ),
                    np.asarray(
                        category_scores
                    )
                )
            )

        else:

            category_auc = None

        results[category] = {

            "development_defect_images":
                len(defect_entries),

            "valid_images":
                valid_images,

            "missing_masks":
                missing_masks,

            "pixel_auroc":
                category_auc,

            "feature_map": [
                FEATURE_H,
                FEATURE_W
            ],

            "descriptor_dimension":
                FEATURE_C,

            "reference_bank_images":
                int(
                    spatial_bank.shape[0]
                ),

            "matching_strategy":
                "same_spatial_location",
        }

        if category_auc is not None:

            print(
                f"Pixel AUROC: "
                f"{category_auc:.4f}"
            )

        else:

            print(
                "Pixel AUROC: undefined"
            )

    # ========================================================
    # MACRO PIXEL AUROC
    # ========================================================

    valid_aucs = [

        result["pixel_auroc"]

        for result in results.values()

        if result["pixel_auroc"] is not None
    ]

    if len(valid_aucs) > 0:

        macro_pixel_auroc = float(
            np.mean(
                valid_aucs
            )
        )

    else:

        macro_pixel_auroc = None

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {

        "protocol": {

            "model":
                "MobileNetV3-Small",

            "representation":
                "L8",

            "feature_map":
                "48x14x14",

            "patch_count":
                196,

            "descriptor_dimension":
                48,

            "distance":
                "1 - cosine_similarity",

            "reference_bank":
                "canonical normal reference split",

            "evaluation_split":
                "development defect split",

            "matching_strategy":
                "same spatial location",

            "heatmap_interpolation":
                "bicubic",

            "final_test_used":
                False,
        },

        "baseline_comparison":
            {
                "global_reference_macro_pixel_auroc":
                    0.9041
            },

        "results":
            results,

        "macro_pixel_auroc":
            macro_pixel_auroc,
    }

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)

    if macro_pixel_auroc is not None:

        print(
            f"MACRO PIXEL AUROC: "
            f"{macro_pixel_auroc:.4f}"
        )

        print()
        print(
            "GLOBAL REFERENCE BASELINE: "
            "0.9041"
        )

        print(
            f"SPATIAL - GLOBAL: "
            f"{macro_pixel_auroc - 0.9041:+.4f}"
        )

    else:

        print(
            "MACRO PIXEL AUROC: undefined"
        )

    print("=" * 72)

    print(
        f"Saved: {OUTPUT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()