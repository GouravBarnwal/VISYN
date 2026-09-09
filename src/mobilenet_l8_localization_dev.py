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

SPLITS_PATH = Path("artifacts/splits/normal_splits.json")
DEFECT_SPLITS_PATH = Path("artifacts/splits/defect_splits.json")

BANK_DIR = Path("artifacts/patch_banks/mobilenet_l8")

OUTPUT_DIR = Path("artifacts/evaluation/localization")
OUTPUT_PATH = OUTPUT_DIR / "mobilenet_l8_localization_dev.json"

DATA_ROOT = Path("data")

DEVICE = torch.device("cpu")

FEATURE_H = 14
FEATURE_W = 14

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(image_path):
    """
    Convert stored dataset-relative paths such as:

        bottle/test/broken_large/004.png

    into:

        data/bottle/test/broken_large/004.png

    Absolute paths are returned unchanged.
    """

    image_path = Path(image_path)

    if image_path.is_absolute():
        return image_path

    return DATA_ROOT / image_path


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = mobilenet_v3_small(
        weights=weights
    )

    # MobileNetV3-Small L8:
    #
    # model.features[:9]
    #
    # output:
    # [B, 48, 14, 14]

    feature_extractor = nn.Sequential(
        *list(model.features.children())[:9]
    )

    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    return feature_extractor


# ============================================================
# LOAD PATCH BANK
# ============================================================

def load_bank(category):

    bank_path = BANK_DIR / f"{category}.npy"

    if not bank_path.exists():
        raise FileNotFoundError(
            f"Reference bank not found: {bank_path}"
        )

    bank = np.load(
        bank_path
    ).astype(
        np.float32
    )

    # Defensive L2 normalization.
    norms = np.linalg.norm(
        bank,
        axis=1,
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

    # image_path is expected to already be resolved.
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
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
    )

    feature_map = feature_map.squeeze(
        0
    )

    # Expected:
    # [48, 14, 14]

    if feature_map.ndim != 3:
        raise RuntimeError(
            f"Unexpected feature map dimensions: "
            f"{feature_map.shape}"
        )

    if feature_map.shape[0] != 48:
        raise RuntimeError(
            f"Unexpected feature channels: "
            f"{feature_map.shape}"
        )

    if feature_map.shape[1] != FEATURE_H:
        raise RuntimeError(
            f"Unexpected feature height: "
            f"{feature_map.shape}"
        )

    if feature_map.shape[2] != FEATURE_W:
        raise RuntimeError(
            f"Unexpected feature width: "
            f"{feature_map.shape}"
        )

    return feature_map.cpu().numpy()


# ============================================================
# SPATIAL PATCH DESCRIPTORS
# ============================================================

def feature_map_to_patches(
    feature_map
):

    # [C, H, W]
    c, h, w = feature_map.shape

    # [H, W, C]
    spatial = np.transpose(
        feature_map,
        (1, 2, 0)
    )

    # [H*W, C]
    patches = spatial.reshape(
        h * w,
        c
    )

    # L2 normalize each spatial descriptor.
    norms = np.linalg.norm(
        patches,
        axis=1,
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
# PATCH ANOMALY DISTANCES
# ============================================================

def compute_patch_scores(
    patches,
    reference_bank
):

    # Both patches and reference bank are L2 normalized.
    #
    # cosine similarity:
    #
    #     patch · reference
    #
    # anomaly distance:
    #
    #     1 - max similarity

    similarities = (
        patches @ reference_bank.T
    )

    best_similarity = np.max(
        similarities,
        axis=1
    )

    distances = (
        1.0 - best_similarity
    )

    return distances.astype(
        np.float32
    )


# ============================================================
# HEATMAP RECONSTRUCTION
# ============================================================

def reconstruct_heatmap(
    patch_scores
):

    expected_patches = (
        FEATURE_H * FEATURE_W
    )

    if patch_scores.shape[0] != expected_patches:
        raise RuntimeError(
            f"Expected "
            f"{expected_patches} patch scores, "
            f"got {patch_scores.shape[0]}"
        )

    return patch_scores.reshape(
        FEATURE_H,
        FEATURE_W
    )


# ============================================================
# UPSAMPLE HEATMAP
# ============================================================

def upsample_heatmap(
    heatmap,
    height,
    width
):

    upsampled = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    return upsampled.astype(
        np.float32
    )


# ============================================================
# LOAD GROUND-TRUTH MASK
# ============================================================

def find_mask(
    image_path
):

    image_path = Path(
        image_path
    )

    # Stored path:
    #
    # bottle/test/broken_large/004.png
    #
    # Therefore:
    #
    # parts[0] = bottle
    # parts[1] = test
    # parts[2] = broken_large
    # parts[3] = 004.png

    if len(image_path.parts) < 4:
        return None

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
# MASK LOADING
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

    mask = mask > 0

    return mask


# ============================================================
# PIXEL AUROC
# ============================================================

def pixel_auroc(
    heatmap,
    mask
):

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

    # Pixel AUROC requires both classes.
    if np.unique(labels).size < 2:
        return None

    return float(
        roc_auc_score(
            labels,
            scores
        )
    )


# ============================================================
# COORDINATE SANITY CHECK
# ============================================================

def coordinate_sanity_check():

    # Patch index:
    #
    # index = row * width + col

    index = (
        5 * FEATURE_W + 7
    )

    heatmap = np.zeros(
        (
            FEATURE_H,
            FEATURE_W
        ),
        dtype=np.float32
    )

    heatmap[5, 7] = 1.0

    recovered_index = np.argmax(
        heatmap
    )

    if recovered_index != index:
        raise RuntimeError(
            "Spatial coordinate mapping failed."
        )

    print(
        "Coordinate sanity check: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — MOBILENET L8 LOCALIZATION"
    )
    print("=" * 72)

    coordinate_sanity_check()

    model = build_model()

    print(
        "Model loaded: MobileNetV3-Small L8"
    )

    print(
        f"Feature map: "
        f"{FEATURE_H} x {FEATURE_W}"
    )

    # --------------------------------------------------------
    # Load defect split
    # --------------------------------------------------------

    if not DEFECT_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Defect split not found: "
            f"{DEFECT_SPLITS_PATH}"
        )

    defect_split = json.loads(
        DEFECT_SPLITS_PATH.read_text(
            encoding="utf-8"
        )
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

        # ----------------------------------------------------
        # Reference bank
        # ----------------------------------------------------

        reference_bank = load_bank(
            category
        )

        print(
            f"Reference bank: "
            f"{reference_bank.shape}"
        )

        # ----------------------------------------------------
        # Development defect images
        # ----------------------------------------------------

        defect_entries = (
            defect_split[
                "categories"
            ][
                category
            ][
                "development"
            ]
        )

        category_scores = []
        category_labels = []

        valid_images = 0

        # ====================================================
        # IMAGE LOOP
        # ====================================================

        for image_path in defect_entries:

            image_path = Path(
                image_path
            )

            # ------------------------------------------------
            # Resolve image path ONCE
            # ------------------------------------------------

            resolved_image_path = (
                resolve_image_path(
                    image_path
                )
            )

            # ------------------------------------------------
            # Ground-truth mask
            # ------------------------------------------------

            mask_path = find_mask(
                image_path
            )

            if mask_path is None:

                print(
                    f"WARNING: mask not found: "
                    f"{image_path}"
                )

                continue

            # ------------------------------------------------
            # Verify image exists
            # ------------------------------------------------

            if not resolved_image_path.exists():

                raise FileNotFoundError(
                    f"Image not found: "
                    f"{resolved_image_path}"
                )

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

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
                feature_map_to_patches(
                    feature_map
                )
            )

            # ------------------------------------------------
            # Patch anomaly scores
            # ------------------------------------------------

            patch_scores = (
                compute_patch_scores(
                    patches,
                    reference_bank
                )
            )

            # ------------------------------------------------
            # 14 x 14 anomaly map
            # ------------------------------------------------

            low_res_heatmap = (
                reconstruct_heatmap(
                    patch_scores
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
            # Pixel metrics
            # ------------------------------------------------

            auc = pixel_auroc(
                heatmap,
                mask
            )

            if auc is not None:

                category_scores.extend(
                    heatmap.reshape(
                        -1
                    ).tolist()
                )

                category_labels.extend(
                    mask.reshape(
                        -1
                    ).astype(
                        np.uint8
                    ).tolist()
                )

            valid_images += 1

        # ====================================================
        # CATEGORY-LEVEL PIXEL AUROC
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

        # ----------------------------------------------------
        # Save category result
        # ----------------------------------------------------

        results[category] = {

            "development_defect_images":
                len(defect_entries),

            "valid_images":
                valid_images,

            "pixel_auroc":
                category_auc,

            "feature_map": [
                FEATURE_H,
                FEATURE_W
            ],

            "descriptor_dimension":
                48,

            "reference_bank_size":
                int(
                    reference_bank.shape[0]
                ),
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
    # MACRO AVERAGE
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

            "aggregation":
                "spatial patch scores",

            "reference_bank":
                "canonical normal reference split",

            "evaluation_split":
                "development defect split",

            "final_test_used":
                False,
        },

        "results":
            results,

        "macro_pixel_auroc":
            macro_pixel_auroc,
    }

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 72)

    if macro_pixel_auroc is not None:

        print(
            f"MACRO PIXEL AUROC: "
            f"{macro_pixel_auroc:.4f}"
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