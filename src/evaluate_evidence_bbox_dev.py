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
# HEATMAP
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
        (maximum - minimum)
    ).astype(np.float32)


# ============================================================
# THRESHOLD STRATEGIES
# ============================================================

def threshold_percentile(
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


def threshold_mean_std(
    heatmap,
    multiplier,
):

    threshold = (
        np.mean(heatmap)
        +
        multiplier
        *
        np.std(heatmap)
    )

    return (
        heatmap >= threshold
    ).astype(np.uint8)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def connected_components(
    mask,
):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    components = []

    for label in range(
        1,
        num_labels,
    ):

        x = stats[
            label,
            cv2.CC_STAT_LEFT,
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP,
        ]

        width = stats[
            label,
            cv2.CC_STAT_WIDTH,
        ]

        height = stats[
            label,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        components.append({
            "label": label,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "area": int(area),
            "cx": float(
                centroids[label][0]
            ),
            "cy": float(
                centroids[label][1]
            ),
        })

    return components


# ============================================================
# COMPONENT SELECTION
# ============================================================

def select_component(
    mask,
    heatmap,
    method,
):

    components = (
        connected_components(
            mask
        )
    )

    if not components:

        return None

    # --------------------------------------------------------
    # Largest spatial component
    # --------------------------------------------------------

    if method == "LARGEST":

        return max(
            components,
            key=lambda c: c["area"],
        )

    # --------------------------------------------------------
    # Component with highest average heatmap response
    # --------------------------------------------------------

    if method == "HIGHEST_MEAN_SCORE":

        best = None
        best_score = -np.inf

        for component in components:

            label = component[
                "label"
            ]

            region = (
                mask == label
            )

            score = float(
                np.mean(
                    heatmap[region]
                )
            )

            if score > best_score:

                best_score = score
                best = component

        return best

    # --------------------------------------------------------
    # Component with highest total anomaly energy
    # --------------------------------------------------------

    if method == "HIGHEST_TOTAL_SCORE":

        best = None
        best_score = -np.inf

        for component in components:

            label = component[
                "label"
            ]

            region = (
                mask == label
            )

            score = float(
                np.sum(
                    heatmap[region]
                )
            )

            if score > best_score:

                best_score = score
                best = component

        return best

    raise ValueError(
        f"Unknown component method: {method}"
    )


# ============================================================
# BOX EXPANSION
# ============================================================

def expand_box(
    component,
    feature_width,
    feature_height,
    expansion,
):

    x1 = component["x"]
    y1 = component["y"]

    x2 = (
        component["x"]
        +
        component["width"]
        -
        1
    )

    y2 = (
        component["y"]
        +
        component["height"]
        -
        1
    )

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    width = (
        x2 - x1 + 1
    )

    height = (
        y2 - y1 + 1
    )

    new_width = (
        width
        *
        (1.0 + expansion)
    )

    new_height = (
        height
        *
        (1.0 + expansion)
    )

    new_x1 = (
        center_x
        -
        new_width / 2.0
    )

    new_y1 = (
        center_y
        -
        new_height / 2.0
    )

    new_x2 = (
        center_x
        +
        new_width / 2.0
    )

    new_y2 = (
        center_y
        +
        new_height / 2.0
    )

    new_x1 = max(
        0,
        int(np.floor(new_x1)),
    )

    new_y1 = max(
        0,
        int(np.floor(new_y1)),
    )

    new_x2 = min(
        feature_width - 1,
        int(np.ceil(new_x2)),
    )

    new_y2 = min(
        feature_height - 1,
        int(np.ceil(new_y2)),
    )

    return {
        "x1": new_x1,
        "y1": new_y1,
        "x2": new_x2,
        "y2": new_y2,
    }


# ============================================================
# BOX → IMAGE COORDINATES
# ============================================================

def scale_box_to_image(
    box,
    feature_width,
    feature_height,
    image_width,
    image_height,
):

    x1 = int(
        round(
            box["x1"]
            *
            image_width
            /
            feature_width
        )
    )

    y1 = int(
        round(
            box["y1"]
            *
            image_height
            /
            feature_height
        )
    )

    x2 = int(
        round(
            (
                box["x2"] + 1
            )
            *
            image_width
            /
            feature_width
        )
        - 1
    )

    y2 = int(
        round(
            (
                box["y2"] + 1
            )
            *
            image_height
            /
            feature_height
        )
        - 1
    )

    x1 = max(
        0,
        min(
            x1,
            image_width - 1,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            image_height - 1,
        ),
    )

    x2 = max(
        x1,
        min(
            x2,
            image_width - 1,
        ),
    )

    y2 = max(
        y1,
        min(
            y2,
            image_height - 1,
        ),
    )

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


# ============================================================
# GROUND TRUTH BOX
# ============================================================

def mask_to_box(
    mask,
):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        return None

    return {
        "x1": int(xs.min()),
        "y1": int(ys.min()),
        "x2": int(xs.max()),
        "y2": int(ys.max()),
    }


# ============================================================
# BOX METRICS
# ============================================================

def box_iou(
    predicted,
    ground_truth,
):

    if (
        predicted is None
        or
        ground_truth is None
    ):

        return 0.0

    ix1 = max(
        predicted["x1"],
        ground_truth["x1"],
    )

    iy1 = max(
        predicted["y1"],
        ground_truth["y1"],
    )

    ix2 = min(
        predicted["x2"],
        ground_truth["x2"],
    )

    iy2 = min(
        predicted["y2"],
        ground_truth["y2"],
    )

    if (
        ix2 < ix1
        or
        iy2 < iy1
    ):

        return 0.0

    intersection = (
        ix2 - ix1 + 1
    ) * (
        iy2 - iy1 + 1
    )

    predicted_area = (
        predicted["x2"]
        -
        predicted["x1"]
        + 1
    ) * (
        predicted["y2"]
        -
        predicted["y1"]
        + 1
    )

    ground_truth_area = (
        ground_truth["x2"]
        -
        ground_truth["x1"]
        + 1
    ) * (
        ground_truth["y2"]
        -
        ground_truth["y1"]
        + 1
    )

    union = (
        predicted_area
        +
        ground_truth_area
        -
        intersection
    )

    return float(
        intersection / union
    )


def ground_truth_box_coverage(
    predicted,
    ground_truth,
):

    if (
        predicted is None
        or
        ground_truth is None
    ):

        return 0.0

    ix1 = max(
        predicted["x1"],
        ground_truth["x1"],
    )

    iy1 = max(
        predicted["y1"],
        ground_truth["y1"],
    )

    ix2 = min(
        predicted["x2"],
        ground_truth["x2"],
    )

    iy2 = min(
        predicted["y2"],
        ground_truth["y2"],
    )

    if (
        ix2 < ix1
        or
        iy2 < iy1
    ):

        return 0.0

    intersection = (
        ix2 - ix1 + 1
    ) * (
        iy2 - iy1 + 1
    )

    ground_truth_area = (
        ground_truth["x2"]
        -
        ground_truth["x1"]
        + 1
    ) * (
        ground_truth["y2"]
        -
        ground_truth["y1"]
        + 1
    )

    return float(
        intersection
        /
        ground_truth_area
    )


# ============================================================
# FIND MASK
# ============================================================

def find_mask(
    relative_path,
):

    relative_path = Path(
        relative_path
    )

    category = (
        relative_path.parts[0]
    )

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
        "Unsupported development split format"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — LOCALIZATION "
        "BOUNDING-BOX EVIDENCE — DEVELOPMENT SET"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    with open(
        DEFECT_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        split_data = json.load(f)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MobileNetL8()
    model.eval()

    # --------------------------------------------------------
    # Evidence configurations
    # --------------------------------------------------------

    configurations = [

        (
            "P90_LARGEST",
            lambda h:
            threshold_percentile(
                h,
                90,
            ),
            "LARGEST",
            0.00,
        ),

        (
            "P90_HIGHEST_MEAN",
            lambda h:
            threshold_percentile(
                h,
                90,
            ),
            "HIGHEST_MEAN_SCORE",
            0.00,
        ),

        (
            "P90_HIGHEST_TOTAL",
            lambda h:
            threshold_percentile(
                h,
                90,
            ),
            "HIGHEST_TOTAL_SCORE",
            0.00,
        ),

        (
            "P95_LARGEST",
            lambda h:
            threshold_percentile(
                h,
                95,
            ),
            "LARGEST",
            0.00,
        ),

        (
            "P95_HIGHEST_MEAN",
            lambda h:
            threshold_percentile(
                h,
                95,
            ),
            "HIGHEST_MEAN_SCORE",
            0.00,
        ),

        (
            "MEAN_1STD_LARGEST",
            lambda h:
            threshold_mean_std(
                h,
                1.0,
            ),
            "LARGEST",
            0.00,
        ),

        (
            "MEAN_1_5STD_LARGEST",
            lambda h:
            threshold_mean_std(
                h,
                1.5,
            ),
            "LARGEST",
            0.00,
        ),

        (
            "MEAN_1_5STD_LARGEST_EXPAND",
            lambda h:
            threshold_mean_std(
                h,
                1.5,
            ),
            "LARGEST",
            0.50,
        ),

        (
            "P90_LARGEST_EXPAND",
            lambda h:
            threshold_percentile(
                h,
                90,
            ),
            "LARGEST",
            0.50,
        ),
    ]

    all_results = {}

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

        global_bank = np.load(
            GLOBAL_BANK_ROOT
            / f"{category}.npy"
        )

        spatial_bank = np.load(
            SPATIAL_BANK_ROOT
            / f"{category}.npy"
        )

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

        sample_data = []

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

            if (
                not image_path.exists()
                or
                mask_path is None
            ):

                continue

            image = Image.open(
                image_path
            )

            image_width, image_height = (
                image.size
            )

            ground_truth = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if ground_truth is None:

                continue

            ground_truth = (
                ground_truth > 0
            ).astype(np.uint8)

            feature_map = (
                extract_feature_map(
                    model,
                    image_path,
                )
            )

            global_map = (
                global_heatmap(
                    feature_map,
                    global_bank,
                )
            )

            spatial_map = (
                spatial_heatmap(
                    feature_map,
                    spatial_bank,
                )
            )

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

            sample_data.append({
                "heatmap": hybrid_map,
                "ground_truth_box":
                    mask_to_box(
                        ground_truth
                    ),
                "image_width":
                    image_width,
                "image_height":
                    image_height,
            })

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
            f"{len(sample_data)}"
        )

        # ----------------------------------------------------
        # Evaluate configurations
        # ----------------------------------------------------

        results = []

        for (
            name,
            threshold_fn,
            component_method,
            expansion,
        ) in configurations:

            ious = []
            coverages = []
            box_widths = []
            box_heights = []
            empty_count = 0

            for sample in sample_data:

                heatmap = sample[
                    "heatmap"
                ]

                mask = threshold_fn(
                    heatmap
                )

                component = (
                    select_component(
                        mask,
                        heatmap,
                        component_method,
                    )
                )

                if component is None:

                    empty_count += 1
                    ious.append(0.0)
                    coverages.append(0.0)
                    continue

                feature_height, feature_width = (
                    heatmap.shape
                )

                box_feature = (
                    expand_box(
                        component,
                        feature_width,
                        feature_height,
                        expansion,
                    )
                )

                predicted_box = (
                    scale_box_to_image(
                        box_feature,
                        feature_width,
                        feature_height,
                        sample[
                            "image_width"
                        ],
                        sample[
                            "image_height"
                        ],
                    )
                )

                truth_box = sample[
                    "ground_truth_box"
                ]

                iou = box_iou(
                    predicted_box,
                    truth_box,
                )

                coverage = (
                    ground_truth_box_coverage(
                        predicted_box,
                        truth_box,
                    )
                )

                ious.append(
                    iou
                )

                coverages.append(
                    coverage
                )

                width = (
                    predicted_box["x2"]
                    -
                    predicted_box["x1"]
                    +
                    1
                )

                height = (
                    predicted_box["y2"]
                    -
                    predicted_box["y1"]
                    +
                    1
                )

                box_widths.append(
                    width
                    /
                    sample[
                        "image_width"
                    ]
                )

                box_heights.append(
                    height
                    /
                    sample[
                        "image_height"
                    ]
                )

            results.append({
                "name": name,
                "box_iou": float(
                    np.mean(ious)
                ),
                "coverage": float(
                    np.mean(coverages)
                ),
                "box_area_ratio": float(
                    np.mean(
                        np.array(
                            box_widths
                        )
                        *
                        np.array(
                            box_heights
                        )
                    )
                ),
                "empty_rate": float(
                    empty_count
                    /
                    len(sample_data)
                ),
            })

        all_results[
            category
        ] = results

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()

        for result in results:

            print(
                f"{result['name']:32s} "
                f"BoxIoU="
                f"{result['box_iou']:.4f}  "
                f"GT_Coverage="
                f"{result['coverage']:.4f}  "
                f"BoxArea="
                f"{result['box_area_ratio']:.4f}  "
                f"Empty="
                f"{result['empty_rate']:.4f}"
            )

    # ========================================================
    # MACRO
    # ========================================================

    print()
    print("=" * 72)
    print(
        "MACRO RESULTS"
    )
    print("=" * 72)

    for name, _, _, _ in configurations:

        rows = []

        for category in CATEGORIES:

            for result in (
                all_results[
                    category
                ]
            ):

                if (
                    result["name"]
                    == name
                ):

                    rows.append(
                        result
                    )

        print(
            f"{name:32s} "
            f"BoxIoU="
            f"{np.mean([r['box_iou'] for r in rows]):.4f}  "
            f"GT_Coverage="
            f"{np.mean([r['coverage'] for r in rows]):.4f}  "
            f"BoxArea="
            f"{np.mean([r['box_area_ratio'] for r in rows]):.4f}  "
            f"Empty="
            f"{np.mean([r['empty_rate'] for r in rows]):.4f}"
        )

    print()
    print("=" * 72)
    print(
        "BOUNDING-BOX EVIDENCE EXPERIMENT COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()