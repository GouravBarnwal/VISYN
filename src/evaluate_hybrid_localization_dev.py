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

GLOBAL_BANK_DIR = Path(
    "artifacts/patch_banks/mobilenet_l8"
)

SPATIAL_BANK_DIR = Path(
    "artifacts/patch_banks/mobilenet_l8_spatial"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/localization"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mobilenet_l8_hybrid_localization_dev.json"
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

# alpha = contribution of spatial localization
#
# 0.00 = global only
# 0.25 = 25% spatial + 75% global
# 0.50 = 50% spatial + 50% global
# 0.75 = 75% spatial + 25% global
# 1.00 = spatial only

ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
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
# LOAD GLOBAL BANK
# ============================================================

def load_global_bank(category):

    bank_path = (
        GLOBAL_BANK_DIR
        / f"{category}.npy"
    )

    if not bank_path.exists():

        raise FileNotFoundError(
            f"Global bank not found: "
            f"{bank_path}"
        )

    bank = np.load(
        bank_path
    ).astype(
        np.float32
    )

    if bank.ndim != 2:

        raise RuntimeError(
            f"Unexpected global bank shape: "
            f"{bank.shape}"
        )

    if bank.shape[1] != FEATURE_C:

        raise RuntimeError(
            f"Unexpected global descriptor dimension: "
            f"{bank.shape}"
        )

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
# LOAD SPATIAL BANK
# ============================================================

def load_spatial_bank(category):

    bank_path = (
        SPATIAL_BANK_DIR
        / f"{category}.npy"
    )

    if not bank_path.exists():

        raise FileNotFoundError(
            f"Spatial bank not found: "
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
            f"Unexpected spatial bank dimensions: "
            f"{bank.shape}"
        )

    if bank.shape[1:] != expected_shape[1:]:

        raise RuntimeError(
            f"Unexpected spatial bank shape: "
            f"{bank.shape}"
        )

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
# SPATIAL DESCRIPTORS
# ============================================================

def feature_map_to_spatial_patches(
    feature_map
):

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
# GLOBAL HEATMAP
# ============================================================

def compute_global_heatmap(
    patches,
    global_bank
):

    query = patches.reshape(
        FEATURE_H * FEATURE_W,
        FEATURE_C
    )

    similarities = (
        query @ global_bank.T
    )

    best_similarity = np.max(
        similarities,
        axis=1
    )

    scores = (
        1.0 - best_similarity
    )

    return scores.reshape(
        FEATURE_H,
        FEATURE_W
    ).astype(
        np.float32
    )


# ============================================================
# SPATIAL HEATMAP
# ============================================================

def compute_spatial_heatmap(
    patches,
    spatial_bank
):

    heatmap = np.empty(
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
                spatial_bank[
                    :,
                    row,
                    col,
                    :
                ]
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
# UPSAMPLE
# ============================================================

def upsample_heatmap(
    heatmap,
    height,
    width
):

    return cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    ).astype(
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
# CATEGORY AUROC
# ============================================================

def calculate_auc(
    scores,
    labels
):

    scores = np.concatenate(
        scores
    )

    labels = np.concatenate(
        labels
    )

    if np.unique(labels).size < 2:
        return None

    return float(
        roc_auc_score(
            labels,
            scores
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — HYBRID MOBILENET L8 LOCALIZATION"
    )
    print("=" * 72)

    print(
        "Representation: MobileNetV3-Small L8"
    )

    print(
        "Feature map: 48 x 14 x 14"
    )

    print(
        "Global + spatial reference matching"
    )

    print(
        "Evaluation: development defect split"
    )

    print(
        "Final test used: False"
    )

    print()
    print(
        "Alpha = spatial contribution"
    )

    for alpha in ALPHAS:

        print(
            f"  alpha={alpha:.2f}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    print()
    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Defect split
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

    # ========================================================
    # RESULTS
    # ========================================================

    results = {}

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(category.upper())
        print("-" * 72)

        global_bank = load_global_bank(
            category
        )

        spatial_bank = load_spatial_bank(
            category
        )

        print(
            f"Global bank: "
            f"{global_bank.shape}"
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

        # ----------------------------------------------------
        # Per-alpha storage for THIS category only.
        #
        # NumPy arrays are substantially more memory efficient
        # than giant Python lists.
        # ----------------------------------------------------

        alpha_scores = {
            alpha: []
            for alpha in ALPHAS
        }

        alpha_labels = {
            alpha: []
            for alpha in ALPHAS
        }

        valid_images = 0
        missing_masks = 0

        # ====================================================
        # IMAGE LOOP
        # ====================================================

        for image_index, image_path in enumerate(
            defect_entries,
            start=1
        ):

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
            # Global anomaly map
            # ------------------------------------------------

            global_low_res = (
                compute_global_heatmap(
                    patches,
                    global_bank
                )
            )

            # ------------------------------------------------
            # Spatial anomaly map
            # ------------------------------------------------

            spatial_low_res = (
                compute_spatial_heatmap(
                    patches,
                    spatial_bank
                )
            )

            # ------------------------------------------------
            # Upsample
            # ------------------------------------------------

            global_heatmap = (
                upsample_heatmap(
                    global_low_res,
                    height,
                    width
                )
            )

            spatial_heatmap = (
                upsample_heatmap(
                    spatial_low_res,
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

            labels = mask.reshape(
                -1
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # Hybrid alpha sweep
            # ------------------------------------------------

            for alpha in ALPHAS:

                hybrid_heatmap = (
                    (1.0 - alpha)
                    * global_heatmap
                    +
                    alpha
                    * spatial_heatmap
                )

                scores = (
                    hybrid_heatmap
                    .reshape(-1)
                    .astype(np.float64)
                )

                alpha_scores[
                    alpha
                ].append(
                    scores
                )

                alpha_labels[
                    alpha
                ].append(
                    labels
                )

            valid_images += 1

            if (
                image_index % 10 == 0
                or image_index == len(defect_entries)
            ):

                print(
                    f"Processed "
                    f"{image_index}/"
                    f"{len(defect_entries)}"
                )

        # ====================================================
        # CATEGORY RESULTS
        # ====================================================

        category_results = {}

        print()
        print(
            "Category alpha sweep:"
        )

        for alpha in ALPHAS:

            auc = calculate_auc(
                alpha_scores[alpha],
                alpha_labels[alpha]
            )

            category_results[
                str(alpha)
            ] = auc

            if auc is None:

                print(
                    f"  alpha={alpha:.2f}: "
                    f"undefined"
                )

            else:

                print(
                    f"  alpha={alpha:.2f}: "
                    f"{auc:.4f}"
                )

        results[category] = {

            "development_defect_images":
                len(defect_entries),

            "valid_images":
                valid_images,

            "missing_masks":
                missing_masks,

            "pixel_auroc_by_alpha":
                category_results,
        }

    # ========================================================
    # MACRO RESULTS
    # ========================================================

    macro_results = {}

    print()
    print("=" * 72)
    print(
        "MACRO PIXEL AUROC BY ALPHA"
    )
    print("=" * 72)

    for alpha in ALPHAS:

        category_aucs = {}

        for category in CATEGORIES:

            auc = results[
                category
            ][
                "pixel_auroc_by_alpha"
            ][
                str(alpha)
            ]

            category_aucs[
                category
            ] = auc

        valid_aucs = [
            auc
            for auc in category_aucs.values()
            if auc is not None
        ]

        if len(valid_aucs) > 0:

            macro_auc = float(
                np.mean(
                    valid_aucs
                )
            )

        else:

            macro_auc = None

        macro_results[
            str(alpha)
        ] = {

            "macro_pixel_auroc":
                macro_auc,

            "category_pixel_auroc":
                category_aucs,
        }

        if macro_auc is None:

            print(
                f"alpha={alpha:.2f}: "
                f"undefined"
            )

        else:

            print(
                f"alpha={alpha:.2f}: "
                f"{macro_auc:.4f}"
            )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    valid_alpha_results = [

        (
            alpha,
            result["macro_pixel_auroc"]
        )

        for alpha, result
        in macro_results.items()

        if result["macro_pixel_auroc"]
        is not None
    ]

    if len(valid_alpha_results) > 0:

        best_alpha, best_macro_auc = max(
            valid_alpha_results,
            key=lambda item: item[1]
        )

    else:

        best_alpha = None
        best_macro_auc = None

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

            "evaluation_split":
                "development defect split",

            "final_test_used":
                False,

            "global_reference_bank":
                "canonical normal reference split",

            "spatial_reference_bank":
                "canonical normal reference split",

            "global_matching":
                "all normal reference patches",

            "spatial_matching":
                "same spatial location",

            "hybrid_definition":
                "(1-alpha)*global + alpha*spatial",

            "alpha_definition":
                "spatial contribution",

            "alphas_tested":
                ALPHAS,

            "heatmap_interpolation":
                "bicubic",
        },

        "baseline_results": {

            "global_macro_pixel_auroc":
                0.9041,

            "spatial_macro_pixel_auroc":
                0.9390,
        },

        "category_results":
            results,

        "macro_results":
            macro_results,

        "best_configuration": {

            "alpha":
                float(best_alpha)
                if best_alpha is not None
                else None,

            "macro_pixel_auroc":
                best_macro_auc,
        },
    }

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

    if best_alpha is not None:

        print(
            f"BEST ALPHA: "
            f"{float(best_alpha):.2f}"
        )

        print(
            f"BEST MACRO PIXEL AUROC: "
            f"{best_macro_auc:.4f}"
        )

        print()
        print(
            "GLOBAL BASELINE: 0.9041"
        )

        print(
            "SPATIAL BASELINE: 0.9390"
        )

        print(
            f"HYBRID GAIN VS GLOBAL: "
            f"{best_macro_auc - 0.9041:+.4f}"
        )

        print(
            f"HYBRID GAIN VS SPATIAL: "
            f"{best_macro_auc - 0.9390:+.4f}"
        )

    else:

        print(
            "BEST CONFIGURATION: undefined"
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