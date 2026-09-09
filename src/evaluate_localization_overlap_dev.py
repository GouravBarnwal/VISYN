from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
    "mobilenet_l8_hybrid_overlap_dev.json"
)

ALPHA = 0.75

TOP_PERCENTAGES = [
    0.01,
    0.02,
    0.05,
]

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
        1,
        2,
        0,
    )

    feature_map = feature_map.cpu().numpy()

    # L2-normalize every spatial descriptor.
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
# GLOBAL MATCHING
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
# SPATIAL MATCHING
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
# HYBRID
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
# LOAD MASK
# ============================================================

def load_mask(mask_path):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:

        raise RuntimeError(
            f"Could not read mask: {mask_path}"
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

    return cv2.resize(
        heatmap,
        output_size,
        interpolation=cv2.INTER_CUBIC,
    )


# ============================================================
# TOP-K PIXEL MASK
# ============================================================

def top_percent_mask(
    heatmap,
    percentage,
):

    flat = heatmap.reshape(-1)

    n_pixels = flat.size

    k = max(
        1,
        int(
            np.ceil(
                n_pixels
                * percentage
            )
        ),
    )

    # Find the kth-largest threshold without
    # sorting the entire image.
    threshold_index = (
        n_pixels - k
    )

    threshold = np.partition(
        flat,
        threshold_index,
    )[threshold_index]

    predicted = (
        heatmap >= threshold
    ).astype(np.uint8)

    return predicted


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    predicted,
    ground_truth,
):

    predicted = predicted.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        predicted,
        ground_truth,
    ).sum()

    union = np.logical_or(
        predicted,
        ground_truth,
    ).sum()

    if union == 0:

        return None

    return float(
        intersection / union
    )


# ============================================================
# DICE
# ============================================================

def calculate_dice(
    predicted,
    ground_truth,
):

    predicted = predicted.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        predicted,
        ground_truth,
    ).sum()

    denominator = (
        predicted.sum()
        + ground_truth.sum()
    )

    if denominator == 0:

        return None

    return float(
        (2.0 * intersection)
        / denominator
    )


# ============================================================
# RECALL / DEFECT COVERAGE
# ============================================================

def calculate_coverage(
    predicted,
    ground_truth,
):

    predicted = predicted.astype(bool)
    ground_truth = ground_truth.astype(bool)

    defect_pixels = ground_truth.sum()

    if defect_pixels == 0:

        return None

    covered = np.logical_and(
        predicted,
        ground_truth,
    ).sum()

    return float(
        covered / defect_pixels
    )


# ============================================================
# BOUNDING BOX
# ============================================================

def bounding_box(
    mask,
):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def bounding_box_coverage(
    predicted,
    ground_truth,
):

    gt_box = bounding_box(
        ground_truth
    )

    if gt_box is None:

        return None

    pred_box = bounding_box(
        predicted
    )

    if pred_box is None:

        return 0.0

    gx1, gy1, gx2, gy2 = gt_box
    px1, py1, px2, py2 = pred_box

    ix1 = max(
        gx1,
        px1,
    )

    iy1 = max(
        gy1,
        py1,
    )

    ix2 = min(
        gx2,
        px2,
    )

    iy2 = min(
        gy2,
        py2,
    )

    if ix2 < ix1 or iy2 < iy1:

        return 0.0

    intersection = (
        (ix2 - ix1 + 1)
        * (iy2 - iy1 + 1)
    )

    gt_area = (
        (gx2 - gx1 + 1)
        * (gy2 - gy1 + 1)
    )

    if gt_area == 0:

        return None

    return float(
        intersection / gt_area
    )


# ============================================================
# SAFE MEAN
# ============================================================

def safe_mean(values):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:

        return None

    return float(
        np.mean(values)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — LOCALIZATION OVERLAP ANALYSIS"
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
        # BANKS
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
            f"Global bank: "
            f"{global_bank.shape}"
        )

        print(
            f"Spatial bank: "
            f"{spatial_bank.shape}"
        )

        development = categories_data[
            category
        ]["development"]

        print(
            f"Development defects: "
            f"{len(development)}"
        )

        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        category_metrics = {
            "top_1_percent": {
                "iou": [],
                "dice": [],
                "coverage": [],
                "bbox_coverage": [],
            },
            "top_2_percent": {
                "iou": [],
                "dice": [],
                "coverage": [],
                "bbox_coverage": [],
            },
            "top_5_percent": {
                "iou": [],
                "dice": [],
                "coverage": [],
                "bbox_coverage": [],
            },
        }

        valid_images = 0
        missing_masks = 0

        # ====================================================
        # IMAGE LOOP
        # ====================================================

        for index, image_path in enumerate(
            development,
            start=1,
        ):

            mask_path = find_mask(
                image_path
            )

            if mask_path is None:

                missing_masks += 1

                continue

            # ----------------------------------------------
            # FEATURES
            # ----------------------------------------------

            feature_map = (
                extract_feature_map(
                    model,
                    image_path,
                )
            )

            # ----------------------------------------------
            # GLOBAL
            # ----------------------------------------------

            global_map = (
                global_heatmap(
                    feature_map,
                    global_bank,
                )
            )

            # ----------------------------------------------
            # SPATIAL
            # ----------------------------------------------

            spatial_map = (
                spatial_heatmap(
                    feature_map,
                    spatial_bank,
                )
            )

            # ----------------------------------------------
            # HYBRID
            # ----------------------------------------------

            hybrid_map = (
                hybrid_heatmap(
                    global_map,
                    spatial_map,
                    ALPHA,
                )
            )

            # ----------------------------------------------
            # ORIGINAL SIZE
            # ----------------------------------------------

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

            # ----------------------------------------------
            # GROUND TRUTH
            # ----------------------------------------------

            ground_truth = load_mask(
                mask_path
            )

            if ground_truth.shape != (
                height,
                width,
            ):

                ground_truth = cv2.resize(
                    ground_truth,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # ----------------------------------------------
            # TOP-K METRICS
            # ----------------------------------------------

            for percentage in TOP_PERCENTAGES:

                predicted = (
                    top_percent_mask(
                        hybrid_full,
                        percentage,
                    )
                )

                iou = calculate_iou(
                    predicted,
                    ground_truth,
                )

                dice = calculate_dice(
                    predicted,
                    ground_truth,
                )

                coverage = calculate_coverage(
                    predicted,
                    ground_truth,
                )

                bbox_cov = (
                    bounding_box_coverage(
                        predicted,
                        ground_truth,
                    )
                )

                key = (
                    f"top_"
                    f"{int(percentage * 100)}"
                    f"_percent"
                )

                category_metrics[
                    key
                ]["iou"].append(
                    iou
                )

                category_metrics[
                    key
                ]["dice"].append(
                    dice
                )

                category_metrics[
                    key
                ]["coverage"].append(
                    coverage
                )

                category_metrics[
                    key
                ]["bbox_coverage"].append(
                    bbox_cov
                )

            valid_images += 1

            if (
                index % 10 == 0
                or index == len(development)
            ):

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(development)}"
                )

            # Release large arrays.
            del feature_map
            del global_map
            del spatial_map
            del hybrid_map
            del hybrid_full
            del ground_truth

        # ====================================================
        # CATEGORY SUMMARY
        # ====================================================

        summarized = {}

        for key, metrics in (
            category_metrics.items()
        ):

            summarized[key] = {
                metric_name:
                    safe_mean(values)
                for metric_name, values
                in metrics.items()
            }

        all_results[
            category
        ] = {
            "images": len(development),
            "valid_images": valid_images,
            "missing_masks": missing_masks,
            "metrics": summarized,
        }

        print()
        print(
            f"{category.upper()} SUMMARY"
        )

        for key, values in (
            summarized.items()
        ):

            print(
                f"  {key}: "
                f"IoU={values['iou']:.4f}, "
                f"Dice={values['dice']:.4f}, "
                f"Coverage={values['coverage']:.4f}, "
                f"BBox={values['bbox_coverage']:.4f}"
            )

        print()

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print("=" * 72)
    print(
        "OVERALL LOCALIZATION OVERLAP SUMMARY"
    )
    print("=" * 72)

    overall = {}

    for percentage in TOP_PERCENTAGES:

        key = (
            f"top_"
            f"{int(percentage * 100)}"
            f"_percent"
        )

        iou_values = []
        dice_values = []
        coverage_values = []
        bbox_values = []

        for category_result in (
            all_results.values()
        ):

            metrics = category_result[
                "metrics"
            ][key]

            if metrics["iou"] is not None:
                iou_values.append(
                    metrics["iou"]
                )

            if metrics["dice"] is not None:
                dice_values.append(
                    metrics["dice"]
                )

            if metrics["coverage"] is not None:
                coverage_values.append(
                    metrics["coverage"]
                )

            if (
                metrics["bbox_coverage"]
                is not None
            ):

                bbox_values.append(
                    metrics["bbox_coverage"]
                )

        overall[key] = {
            "macro_iou":
                safe_mean(iou_values),
            "macro_dice":
                safe_mean(dice_values),
            "macro_coverage":
                safe_mean(
                    coverage_values
                ),
            "macro_bbox_coverage":
                safe_mean(
                    bbox_values
                ),
        }

        print()
        print(
            key
        )

        print(
            f"  Macro IoU: "
            f"{overall[key]['macro_iou']:.4f}"
        )

        print(
            f"  Macro Dice: "
            f"{overall[key]['macro_dice']:.4f}"
        )

        print(
            f"  Macro defect coverage: "
            f"{overall[key]['macro_coverage']:.4f}"
        )

        print(
            f"  Macro bbox coverage: "
            f"{overall[key]['macro_bbox_coverage']:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "protocol": {
            "representation":
                "MobileNetV3-Small L8",
            "feature_map":
                "48x14x14",
            "localization":
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
            "top_percentages":
                TOP_PERCENTAGES,
        },
        "categories":
            all_results,
        "overall":
            overall,
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
    print("=" * 72)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()