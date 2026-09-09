from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

DEFECT_SPLIT_PATH = Path(
    "artifacts/splits/defect_splits.json"
)

GLOBAL_BANK_DIR = Path(
    "artifacts/patch_banks/mobilenet_l8"
)

SPATIAL_BANK_DIR = Path(
    "artifacts/patch_banks/mobilenet_l8_spatial"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/localization/"
    "mobilenet_l8_hybrid_defect_type_dev.json"
)

ALPHA = 0.75

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PATH HELPERS
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
# MODEL
# ============================================================

class MobileNetL8(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        self.features = backbone.features[:9]

    def forward(self, x):

        return self.features(x)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_map(model, image_path):

    image_path = resolve_image_path(
        image_path
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    x = transform(
        image
    ).unsqueeze(0)

    feature_map = model(x)

    feature_map = feature_map.squeeze(0)

    feature_map = feature_map.permute(
        1, 2, 0
    )

    feature_map = feature_map.cpu().numpy()

    # Normalize every spatial descriptor.
    norms = np.linalg.norm(
        feature_map,
        axis=2,
        keepdims=True,
    )

    feature_map = (
        feature_map
        / np.maximum(
            norms,
            1e-12,
        )
    )

    return feature_map


# ============================================================
# GLOBAL HEATMAP
# ============================================================

def global_heatmap(
    feature_map,
    global_bank,
):

    h, w, d = feature_map.shape

    query = feature_map.reshape(
        -1,
        d,
    )

    similarities = (
        query @ global_bank.T
    )

    nearest_similarity = np.max(
        similarities,
        axis=1,
    )

    anomaly = (
        1.0
        - nearest_similarity
    )

    return anomaly.reshape(
        h,
        w,
    )


# ============================================================
# SPATIAL HEATMAP
# ============================================================

def spatial_heatmap(
    feature_map,
    spatial_bank,
):

    h, w, d = feature_map.shape

    heatmap = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for r in range(h):

        for c in range(w):

            query = feature_map[r, c]

            references = (
                spatial_bank[:, r, c, :]
            )

            similarities = (
                references @ query
            )

            best_similarity = np.max(
                similarities
            )

            heatmap[r, c] = (
                1.0
                - best_similarity
            )

    return heatmap


# ============================================================
# HYBRID HEATMAP
# ============================================================

def hybrid_heatmap(
    global_map,
    spatial_map,
    alpha,
):

    return (
        (1.0 - alpha) * global_map
        + alpha * spatial_map
    )


# ============================================================
# MASK
# ============================================================

def load_mask(mask_path):

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = np.asarray(
        mask
    )

    return (
        mask > 0
    ).astype(np.uint8)


# ============================================================
# UPSAMPLE
# ============================================================

def upsample_heatmap(
    heatmap,
    output_size,
):

    import cv2

    return cv2.resize(
        heatmap,
        output_size,
        interpolation=cv2.INTER_CUBIC,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — HYBRID LOCALIZATION "
        "DEFECT-TYPE ANALYSIS"
    )
    print("=" * 72)

    print(
        "Representation: MobileNetV3-Small L8"
    )

    print(
        "Feature map: 48 x 14 x 14"
    )

    print(
        "Localization: 25% global + 75% spatial"
    )

    print(
        f"Alpha: {ALPHA:.2f}"
    )

    print(
        "Evaluation: development defect split"
    )

    print(
        "Final test used: False"
    )

    print()

    # --------------------------------------------------------
    # LOAD SPLIT
    # --------------------------------------------------------

    with open(
        DEFECT_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        defect_splits = json.load(f)

    categories_data = defect_splits[
        "categories"
    ]

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MobileNetL8()
    model.eval()

    print(
        "Model loaded."
    )

    print()

    all_results = {}

    category_auroc_values = []

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in categories_data:

        print("=" * 72)
        print(
            category.upper()
        )
        print("=" * 72)

        # ----------------------------------------------------
        # LOAD BANKS
        # ----------------------------------------------------

        global_bank = np.load(
            GLOBAL_BANK_DIR
            / f"{category}.npy"
        )

        spatial_bank = np.load(
            SPATIAL_BANK_DIR
            / f"{category}.npy"
        )

        print(
            f"Global bank: {global_bank.shape}"
        )

        print(
            f"Spatial bank: {spatial_bank.shape}"
        )

        development = categories_data[
            category
        ]["development"]

        # ----------------------------------------------------
        # GROUP IMAGES BY DEFECT TYPE
        # ----------------------------------------------------

        defect_groups = {}

        for image_path in development:

            image_path = Path(
                image_path
            )

            defect_type = (
                image_path.parent.name
            )

            defect_groups.setdefault(
                defect_type,
                [],
            ).append(
                str(image_path)
            )

        category_results = {}

        # ====================================================
        # DEFECT TYPE LOOP
        # ====================================================

        for defect_type in sorted(
            defect_groups
        ):

            images = defect_groups[
                defect_type
            ]

            print("-" * 72)

            print(
                f"DEFECT TYPE: "
                f"{defect_type}"
            )

            print(
                f"Images: {len(images)}"
            )

            score_chunks = []
            mask_chunks = []

            valid_images = 0
            missing_masks = 0

            # ------------------------------------------------
            # IMAGE LOOP
            # ------------------------------------------------

            for index, image_path in enumerate(
                images,
                start=1,
            ):

                mask_path = find_mask(
                    image_path
                )

                if mask_path is None:

                    missing_masks += 1

                    continue

                # --------------------------------------------
                # FEATURE EXTRACTION
                # --------------------------------------------

                feature_map = (
                    extract_feature_map(
                        model,
                        image_path,
                    )
                )

                # --------------------------------------------
                # GLOBAL
                # --------------------------------------------

                global_map = (
                    global_heatmap(
                        feature_map,
                        global_bank,
                    )
                )

                # --------------------------------------------
                # SPATIAL
                # --------------------------------------------

                spatial_map = (
                    spatial_heatmap(
                        feature_map,
                        spatial_bank,
                    )
                )

                # --------------------------------------------
                # HYBRID
                # --------------------------------------------

                hybrid_map = (
                    hybrid_heatmap(
                        global_map,
                        spatial_map,
                        ALPHA,
                    )
                )

                # --------------------------------------------
                # ORIGINAL IMAGE SIZE
                # --------------------------------------------

                image = Image.open(
                    resolve_image_path(
                        image_path
                    )
                )

                width, height = image.size

                hybrid_full = (
                    upsample_heatmap(
                        hybrid_map,
                        (width, height),
                    )
                )

                # --------------------------------------------
                # MASK
                # --------------------------------------------

                mask = load_mask(
                    mask_path
                )

                if mask.shape != (
                    height,
                    width,
                ):

                    import cv2

                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # --------------------------------------------
                # STORE NUMPY ARRAYS
                # --------------------------------------------

                score_chunks.append(
                    hybrid_full.reshape(-1).astype(
                        np.float32
                    )
                )

                mask_chunks.append(
                    mask.reshape(-1).astype(
                        np.uint8
                    )
                )

                valid_images += 1

                if (
                    index % 10 == 0
                    or index == len(images)
                ):

                    print(
                        f"Processed "
                        f"{index}/{len(images)}"
                    )

                # Explicitly release large arrays.
                del feature_map
                del global_map
                del spatial_map
                del hybrid_map
                del hybrid_full
                del mask

            # ------------------------------------------------
            # DEFECT TYPE RESULT
            # ------------------------------------------------

            if valid_images == 0:

                print(
                    "No valid images."
                )

                category_results[
                    defect_type
                ] = {
                    "pixel_auroc": None,
                    "images": 0,
                    "missing_masks": missing_masks,
                }

                continue

            scores = np.concatenate(
                score_chunks
            )

            masks = np.concatenate(
                mask_chunks
            )

            # Make sure both classes exist.
            unique_classes = np.unique(
                masks
            )

            if len(unique_classes) < 2:

                auroc = None

                print(
                    "AUROC unavailable: "
                    "only one pixel class."
                )

            else:

                auroc = float(
                    roc_auc_score(
                        masks,
                        scores,
                    )
                )

                print(
                    f"Pixel AUROC: "
                    f"{auroc:.4f}"
                )

            print(
                f"Valid images: "
                f"{valid_images}"
            )

            print(
                f"Missing masks: "
                f"{missing_masks}"
            )

            category_results[
                defect_type
            ] = {
                "pixel_auroc": auroc,
                "images": len(images),
                "valid_images": valid_images,
                "missing_masks": missing_masks,
            }

            # Free concatenated arrays before
            # moving to the next defect type.
            del scores
            del masks
            del score_chunks
            del mask_chunks

        # ----------------------------------------------------
        # CATEGORY SUMMARY
        # ----------------------------------------------------

        valid_category_scores = [
            result["pixel_auroc"]
            for result in category_results.values()
            if result["pixel_auroc"] is not None
        ]

        if valid_category_scores:

            category_macro = float(
                np.mean(
                    valid_category_scores
                )
            )

            category_auroc_values.append(
                category_macro
            )

            print()
            print(
                f"{category.upper()} "
                f"DEFECT-TYPE MACRO AUROC: "
                f"{category_macro:.4f}"
            )

        else:

            category_macro = None

        all_results[
            category
        ] = {
            "defect_types": category_results,
            "defect_type_macro_pixel_auroc":
                category_macro,
        }

        print()

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    overall_macro = None

    if category_auroc_values:

        overall_macro = float(
            np.mean(
                category_auroc_values
            )
        )

    print("=" * 72)
    print(
        "DEFECT-TYPE LOCALIZATION SUMMARY"
    )
    print("=" * 72)

    for category, result in all_results.items():

        print()
        print(
            category.upper()
        )

        for defect_type, values in (
            result[
                "defect_types"
            ].items()
        ):

            auroc = values[
                "pixel_auroc"
            ]

            if auroc is None:

                print(
                    f"  {defect_type}: N/A"
                )

            else:

                print(
                    f"  {defect_type}: "
                    f"{auroc:.4f}"
                )

        macro = result[
            "defect_type_macro_pixel_auroc"
        ]

        if macro is not None:

            print(
                f"  MACRO: {macro:.4f}"
            )

    print()
    print("=" * 72)

    if overall_macro is not None:

        print(
            "OVERALL DEFECT-TYPE MACRO "
            f"PIXEL AUROC: {overall_macro:.4f}"
        )

    print("=" * 72)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "protocol": {
            "representation":
                "MobileNetV3-Small L8",
            "feature_map":
                "48x14x14",
            "localization_method":
                "hybrid_global_spatial",
            "alpha":
                ALPHA,
            "global_contribution":
                1.0 - ALPHA,
            "spatial_contribution":
                ALPHA,
            "split":
                "development defect split",
            "final_test_used":
                False,
            "seed":
                SEED,
        },
        "categories":
            all_results,
        "overall_defect_type_macro_pixel_auroc":
            overall_macro,
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
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()