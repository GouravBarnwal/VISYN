import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
    "artifacts/patch_banks/mobilenet_l8"
)

OUTPUT_DIR = Path(
    "artifacts/evaluation/localization/visualizations"
)

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
# MODEL / PREPROCESSING
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()


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
# PATHS
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
# PATCH BANK
# ============================================================

def load_bank(category):

    bank_path = (
        BANK_DIR
        / f"{category}.npy"
    )

    bank = np.load(
        bank_path
    ).astype(
        np.float32
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
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_map(
    model,
    image_path
):

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

    return feature_map.cpu().numpy()


# ============================================================
# PATCH DESCRIPTORS
# ============================================================

def feature_map_to_patches(
    feature_map
):

    c, h, w = feature_map.shape

    spatial = np.transpose(
        feature_map,
        (1, 2, 0)
    )

    patches = spatial.reshape(
        h * w,
        c
    )

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
# PATCH ANOMALY
# ============================================================

def compute_patch_scores(
    patches,
    reference_bank
):

    similarities = (
        patches @ reference_bank.T
    )

    best_similarity = np.max(
        similarities,
        axis=1
    )

    return (
        1.0 - best_similarity
    ).astype(
        np.float32
    )


# ============================================================
# HEATMAP
# ============================================================

def build_heatmap(
    patch_scores,
    height,
    width
):

    low_res = patch_scores.reshape(
        FEATURE_H,
        FEATURE_W
    )

    heatmap = cv2.resize(
        low_res,
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    return heatmap.astype(
        np.float32
    )


# ============================================================
# NORMALIZE HEATMAP FOR VISUALIZATION
# ============================================================

def normalize_for_display(
    heatmap
):

    minimum = float(
        np.min(heatmap)
    )

    maximum = float(
        np.max(heatmap)
    )

    if maximum - minimum < 1e-12:

        return np.zeros_like(
            heatmap,
            dtype=np.uint8
        )

    normalized = (
        (heatmap - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return np.clip(
        normalized,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    image,
    mask,
    heatmap
):

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt = np.zeros_like(
        image_bgr
    )

    gt[mask] = (
        0,
        255,
        0
    )

    ground_truth_overlay = cv2.addWeighted(
        image_bgr,
        0.70,
        gt,
        0.30,
        0
    )

    # --------------------------------------------------------
    # Predicted heatmap
    # --------------------------------------------------------

    heatmap_uint8 = normalize_for_display(
        heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_overlay = cv2.addWeighted(
        image_bgr,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    # --------------------------------------------------------
    # Ground-truth bounding box
    # --------------------------------------------------------

    gt_box = image_bgr.copy()

    ys, xs = np.where(mask)

    if len(xs) > 0:

        x1 = int(np.min(xs))
        x2 = int(np.max(xs))
        y1 = int(np.min(ys))
        y2 = int(np.max(ys))

        cv2.rectangle(
            gt_box,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    # --------------------------------------------------------
    # Predicted heatmap high-response region
    # --------------------------------------------------------

    prediction = image_bgr.copy()

    threshold = np.percentile(
        heatmap,
        95
    )

    predicted_region = (
        heatmap >= threshold
    ).astype(
        np.uint8
    ) * 255

    contours, _ = cv2.findContours(
        predicted_region,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv2.contourArea(contour) < 20:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            prediction,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

    # --------------------------------------------------------
    # Resize panels
    # --------------------------------------------------------

    panel_width = 450
    panel_height = 450

    panels = []

    for panel in [
        image_bgr,
        ground_truth_overlay,
        heatmap_overlay,
        prediction,
    ]:

        panel = cv2.resize(
            panel,
            (
                panel_width,
                panel_height
            ),
            interpolation=cv2.INTER_AREA
        )

        panels.append(panel)

    # --------------------------------------------------------
    # 2 x 2 layout
    # --------------------------------------------------------

    top = np.hstack(
        [
            panels[0],
            panels[1],
        ]
    )

    bottom = np.hstack(
        [
            panels[2],
            panels[3],
        ]
    )

    canvas = np.vstack(
        [
            top,
            bottom,
        ]
    )

    return canvas


# ============================================================
# SELECT REPRESENTATIVE IMAGES
# ============================================================

def select_images(
    defect_entries
):

    # Keep the visualization set manageable.
    #
    # We intentionally choose the first few development
    # examples from each category. This is NOT used for
    # evaluation or model selection.

    return defect_entries[:3]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — LOCALIZATION VISUAL INSPECTION"
    )
    print("=" * 72)

    model = build_model()

    print(
        "Model loaded: MobileNetV3-Small L8"
    )

    defect_split = json.loads(
        DEFECT_SPLITS_PATH.read_text(
            encoding="utf-8"
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_saved = 0

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(category.upper())
        print("-" * 72)

        reference_bank = load_bank(
            category
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

        selected = select_images(
            defect_entries
        )

        category_dir = (
            OUTPUT_DIR / category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_path in selected:

            image_path = Path(
                image_path
            )

            resolved_image_path = (
                resolve_image_path(
                    image_path
                )
            )

            mask_path = find_mask(
                image_path
            )

            if mask_path is None:

                print(
                    f"WARNING: mask not found: "
                    f"{image_path}"
                )

                continue

            if not resolved_image_path.exists():

                print(
                    f"WARNING: image not found: "
                    f"{resolved_image_path}"
                )

                continue

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = np.array(
                Image.open(
                    resolved_image_path
                ).convert("RGB")
            )

            height, width = image.shape[:2]

            # ------------------------------------------------
            # Features
            # ------------------------------------------------

            feature_map = (
                extract_feature_map(
                    model,
                    resolved_image_path
                )
            )

            # ------------------------------------------------
            # Patches
            # ------------------------------------------------

            patches = (
                feature_map_to_patches(
                    feature_map
                )
            )

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            patch_scores = (
                compute_patch_scores(
                    patches,
                    reference_bank
                )
            )

            # ------------------------------------------------
            # Heatmap
            # ------------------------------------------------

            heatmap = build_heatmap(
                patch_scores,
                height,
                width
            )

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:

                print(
                    f"WARNING: could not read mask: "
                    f"{mask_path}"
                )

                continue

            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

            mask = mask > 0

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            canvas = create_visualization(
                image,
                mask,
                heatmap
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_path = (
                category_dir
                / f"{image_path.stem}_localization.jpg"
            )

            cv2.imwrite(
                str(output_path),
                canvas
            )

            print(
                f"Saved: {output_path}"
            )

            total_saved += 1

    print()
    print("=" * 72)
    print(
        f"TOTAL VISUALIZATIONS: {total_saved}"
    )
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()