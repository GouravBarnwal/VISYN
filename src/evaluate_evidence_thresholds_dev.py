from pathlib import Path
import json

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

GLOBAL_BANK_ROOT = Path(
    "artifacts/patch_banks/mobilenet_l8"
)

SPATIAL_BANK_ROOT = Path(
    "artifacts/patch_banks/mobilenet_l8_spatial"
)

ALPHA = 0.75

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]


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
# TRANSFORM
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
def extract_feature_map(
    model,
    image_path,
):

    image = Image.open(
        image_path
    ).convert("RGB")

    x = transform(
        image
    ).unsqueeze(0)

    feature_map = model(x)

    feature_map = (
        feature_map
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    norms = np.linalg.norm(
        feature_map,
        axis=2,
        keepdims=True,
    )

    feature_map = (
        feature_map
        /
        np.maximum(
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
    bank,
):

    h, w, d = feature_map.shape

    query = feature_map.reshape(
        -1,
        d,
    )

    similarities = (
        query @ bank.T
    )

    best_similarity = np.max(
        similarities,
        axis=1,
    )

    anomaly = (
        1.0
        -
        best_similarity
    )

    return anomaly.reshape(
        h,
        w,
    ).astype(np.float32)


# ============================================================
# SPATIAL HEATMAP
# ============================================================

def spatial_heatmap(
    feature_map,
    spatial_bank,
):

    h, w, d = feature_map.shape

    result = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    for r in range(h):

        for c in range(w):

            query = feature_map[
                r,
                c,
            ]

            references = spatial_bank[
                :,
                r,
                c,
                :
            ]

            similarities = (
                references @ query
            )

            result[r, c] = (
                1.0
                -
                np.max(
                    similarities
                )
            )

    return result


# ============================================================
# HYBRID
# ============================================================

def hybrid_heatmap(
    global_map,
    spatial_map,
):

    return (
        (1.0 - ALPHA)
        * global_map
        +
        ALPHA
        * spatial_map
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_heatmap(
    heatmap,
):

    minimum = np.min(
        heatmap
    )

    maximum = np.max(
        heatmap
    )

    if (
        maximum - minimum
        < 1e-12
    ):

        return np.zeros_like(
            heatmap,
            dtype=np.float32,
        )

    return (
        (heatmap - minimum)
        /
        (
            maximum - minimum
        )
    ).astype(np.float32)


# ============================================================
# EVIDENCE THRESHOLDS
# ============================================================

def mask_top_percent(
    heatmap,
    percent,
):

    threshold = np.percentile(
        heatmap,
        100.0 - (
            percent * 100.0
        ),
    )

    return (
        heatmap >= threshold
    ).astype(np.uint8)


def mask_percentile(
    heatmap,
    percentile,
):

    threshold = np.percentile(
        heatmap,
        percentile,
    )

    return (
        heatmap >= threshold
    ).astype(np.uint8)


def mask_mean_std(
    heatmap,
    multiplier,
):

    threshold = (
        np.mean(heatmap)
        +
        multiplier
        * np.std(heatmap)
    )

    return (
        heatmap >= threshold
    ).astype(np.uint8)


# ============================================================
# LIGHT MORPHOLOGY
# ============================================================

def clean_mask(
    mask,
):

    # IMPORTANT:
    # The feature map is only 14x14.
    #
    # A large morphological operation can completely
    # destroy small anomaly regions.
    #
    # Therefore we only perform a very conservative
    # close operation here.

    kernel = np.ones(
        (2, 2),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return mask


# ============================================================
# RESIZE TO ACTUAL GROUND-TRUTH SIZE
# ============================================================

def resize_mask_to_shape(
    mask,
    target_shape,
):

    target_height = target_shape[0]
    target_width = target_shape[1]

    return cv2.resize(
        mask,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )


# ============================================================
# FIND GROUND-TRUTH MASK
# ============================================================

def find_mask(
    relative_path,
):

    relative_path = Path(
        relative_path
    )

    category = relative_path.parts[0]

    defect_type = (
        relative_path.parent.name
    )

    stem = relative_path.stem

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
# METRICS
# ============================================================

def calculate_metrics(
    predicted_mask,
    ground_truth_mask,
):

    predicted = (
        predicted_mask > 0
    )

    ground_truth = (
        ground_truth_mask > 0
    )

    intersection = np.logical_and(
        predicted,
        ground_truth,
    ).sum()

    union = np.logical_or(
        predicted,
        ground_truth,
    ).sum()

    predicted_area = (
        predicted.sum()
    )

    ground_truth_area = (
        ground_truth.sum()
    )

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    dice = (
        2.0 * intersection
        /
        (
            predicted_area
            +
            ground_truth_area
        )
        if (
            predicted_area
            +
            ground_truth_area
        ) > 0
        else 0.0
    )

    coverage = (
        intersection
        /
        ground_truth_area
        if ground_truth_area > 0
        else 0.0
    )

    return (
        float(iou),
        float(dice),
        float(coverage),
    )


# ============================================================
# BOUNDING BOX COVERAGE
# ============================================================

def bounding_box_coverage(
    predicted_mask,
    ground_truth_mask,
):

    gt_y, gt_x = np.where(
        ground_truth_mask > 0
    )

    if len(gt_x) == 0:

        return 0.0

    gt_x1 = gt_x.min()
    gt_x2 = gt_x.max()

    gt_y1 = gt_y.min()
    gt_y2 = gt_y.max()

    pred_y, pred_x = np.where(
        predicted_mask > 0
    )

    if len(pred_x) == 0:

        return 0.0

    pred_x1 = pred_x.min()
    pred_x2 = pred_x.max()

    pred_y1 = pred_y.min()
    pred_y2 = pred_y.max()

    ix1 = max(
        gt_x1,
        pred_x1,
    )

    iy1 = max(
        gt_y1,
        pred_y1,
    )

    ix2 = min(
        gt_x2,
        pred_x2,
    )

    iy2 = min(
        gt_y2,
        pred_y2,
    )

    if (
        ix2 < ix1
        or
        iy2 < iy1
    ):

        return 0.0

    intersection_area = (
        ix2 - ix1 + 1
    ) * (
        iy2 - iy1 + 1
    )

    gt_area = (
        gt_x2 - gt_x1 + 1
    ) * (
        gt_y2 - gt_y1 + 1
    )

    return float(
        intersection_area
        /
        gt_area
    )


# ============================================================
# EVALUATE ONE STRATEGY
# ============================================================

def evaluate_strategy(
    strategy_name,
    predicted_masks,
    ground_truths,
):

    ious = []
    dices = []
    coverages = []
    bbox_coverages = []

    for predicted, truth in zip(
        predicted_masks,
        ground_truths,
    ):

        iou, dice, coverage = (
            calculate_metrics(
                predicted,
                truth,
            )
        )

        bbox = (
            bounding_box_coverage(
                predicted,
                truth,
            )
        )

        ious.append(iou)
        dices.append(dice)
        coverages.append(
            coverage
        )
        bbox_coverages.append(
            bbox
        )

    return {
        "strategy": strategy_name,
        "iou": float(
            np.mean(ious)
        ),
        "dice": float(
            np.mean(dices)
        ),
        "coverage": float(
            np.mean(coverages)
        ),
        "bbox_coverage": float(
            np.mean(
                bbox_coverages
            )
        ),
    }


# ============================================================
# LOAD DEVELOPMENT PATHS
# ============================================================

def load_development_paths(
    split_data,
    category,
):

    development = (
        split_data[
            "categories"
        ][category]["development"]
    )

    if isinstance(
        development,
        list,
    ):

        return development

    if isinstance(
        development,
        dict,
    ):

        paths = []

        for values in (
            development.values()
        ):

            paths.extend(
                values
            )

        return paths

    raise TypeError(
        "Unsupported development split format: "
        f"{type(development)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — EVIDENCE THRESHOLD "
        "REFINEMENT — DEVELOPMENT SET"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load defect split
    # --------------------------------------------------------

    with open(
        DEFECT_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        split_data = json.load(f)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = MobileNetL8()
    model.eval()

    # --------------------------------------------------------
    # Strategies
    # --------------------------------------------------------

    strategies = {

        "TOP_5_PERCENT":
            lambda h:
            mask_top_percent(
                h,
                0.05,
            ),

        "TOP_2_PERCENT":
            lambda h:
            mask_top_percent(
                h,
                0.02,
            ),

        "TOP_1_PERCENT":
            lambda h:
            mask_top_percent(
                h,
                0.01,
            ),

        "P95":
            lambda h:
            mask_percentile(
                h,
                95,
            ),

        "P90":
            lambda h:
            mask_percentile(
                h,
                90,
            ),

        "MEAN_PLUS_1STD":
            lambda h:
            mask_mean_std(
                h,
                1.0,
            ),

        "MEAN_PLUS_1_5STD":
            lambda h:
            mask_mean_std(
                h,
                1.5,
            ),

        "MEAN_PLUS_2STD":
            lambda h:
            mask_mean_std(
                h,
                2.0,
            ),
    }

    # ========================================================
    # RESULTS
    # ========================================================

    category_results = {}

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    for category in CATEGORIES:

        print()
        print("-" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("-" * 72)

        # ----------------------------------------------------
        # Load banks
        # ----------------------------------------------------

        global_bank = np.load(
            GLOBAL_BANK_ROOT
            / f"{category}.npy"
        )

        spatial_bank = np.load(
            SPATIAL_BANK_ROOT
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

        # ----------------------------------------------------
        # Development samples
        # ----------------------------------------------------

        dev_images = (
            load_development_paths(
                split_data,
                category,
            )
        )

        print(
            f"Development defects: "
            f"{len(dev_images)}"
        )

        heatmaps = []
        ground_truths = []

        # ----------------------------------------------------
        # Generate heatmaps
        # ----------------------------------------------------

        for index, relative_path in enumerate(
            dev_images,
            start=1,
        ):

            image_path = (
                DATA_ROOT
                / relative_path
            )

            mask_path = (
                find_mask(
                    relative_path
                )
            )

            if not image_path.exists():

                print(
                    f"WARNING: missing image "
                    f"{image_path}"
                )

                continue

            if mask_path is None:

                print(
                    f"WARNING: missing mask "
                    f"{relative_path}"
                )

                continue

            # ------------------------------------------------
            # Feature map
            # ------------------------------------------------

            feature_map = (
                extract_feature_map(
                    model,
                    image_path,
                )
            )

            # ------------------------------------------------
            # Global anomaly
            # ------------------------------------------------

            global_map = (
                global_heatmap(
                    feature_map,
                    global_bank,
                )
            )

            # ------------------------------------------------
            # Spatial anomaly
            # ------------------------------------------------

            spatial_map = (
                spatial_heatmap(
                    feature_map,
                    spatial_bank,
                )
            )

            # ------------------------------------------------
            # Hybrid anomaly
            # ------------------------------------------------

            hybrid_map = (
                hybrid_heatmap(
                    global_map,
                    spatial_map,
                )
            )

            hybrid_map = (
                normalize_heatmap(
                    hybrid_map
                )
            )

            # ------------------------------------------------
            # Ground truth
            # ------------------------------------------------

            ground_truth = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if ground_truth is None:

                continue

            ground_truth = (
                ground_truth > 0
            ).astype(np.uint8)

            heatmaps.append(
                hybrid_map
            )

            ground_truths.append(
                ground_truth
            )

            if (
                index % 10 == 0
                or
                index == len(dev_images)
            ):

                print(
                    f"Processed "
                    f"{index}/"
                    f"{len(dev_images)}",
                    end="\r",
                )

        print()

        print(
            f"Valid samples: "
            f"{len(heatmaps)}"
        )

        # ----------------------------------------------------
        # Check real image resolution
        # ----------------------------------------------------

        if len(ground_truths) > 0:

            unique_shapes = sorted(
                set(
                    gt.shape
                    for gt in ground_truths
                )
            )

            print(
                f"Ground-truth resolutions: "
                f"{unique_shapes}"
            )

        # ----------------------------------------------------
        # Evaluate each threshold
        # ----------------------------------------------------

        results = []

        for strategy_name, strategy_fn in (
            strategies.items()
        ):

            predicted_masks = []

            for heatmap in heatmaps:

                # --------------------------------------------
                # Threshold at native 14x14 resolution
                # --------------------------------------------

                mask_14 = strategy_fn(
                    heatmap
                )

                # --------------------------------------------
                # Conservative cleanup
                # --------------------------------------------

                mask_14 = clean_mask(
                    mask_14
                )

                # --------------------------------------------
                # Resize according to THIS sample's
                # actual ground-truth resolution.
                # --------------------------------------------

                target_shape = (
                    ground_truths[
                        len(predicted_masks)
                    ].shape
                )

                mask_original = (
                    resize_mask_to_shape(
                        mask_14,
                        target_shape,
                    )
                )

                predicted_masks.append(
                    mask_original
                )

            result = (
                evaluate_strategy(
                    strategy_name,
                    predicted_masks,
                    ground_truths,
                )
            )

            results.append(
                result
            )

        category_results[
            category
        ] = results

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        for result in results:

            print(
                f"{result['strategy']:22s} "
                f"IoU="
                f"{result['iou']:.4f}  "
                f"Dice="
                f"{result['dice']:.4f}  "
                f"Coverage="
                f"{result['coverage']:.4f}  "
                f"BBox="
                f"{result['bbox_coverage']:.4f}"
            )

    # ========================================================
    # MACRO RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print(
        "MACRO RESULTS"
    )
    print("=" * 72)

    for strategy_name in strategies:

        rows = []

        for category in CATEGORIES:

            for result in (
                category_results[
                    category
                ]
            ):

                if (
                    result["strategy"]
                    == strategy_name
                ):

                    rows.append(
                        result
                    )

        print(
            f"{strategy_name:22s} "
            f"IoU="
            f"{np.mean([r['iou'] for r in rows]):.4f}  "
            f"Dice="
            f"{np.mean([r['dice'] for r in rows]):.4f}  "
            f"Coverage="
            f"{np.mean([r['coverage'] for r in rows]):.4f}  "
            f"BBox="
            f"{np.mean([r['bbox_coverage'] for r in rows]):.4f}"
        )

    print()
    print("=" * 72)
    print(
        "EVIDENCE THRESHOLD REFINEMENT COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()