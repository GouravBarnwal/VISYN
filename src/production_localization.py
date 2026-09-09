from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.production_config import (
    CATEGORIES,
    CHUNK_SIZE,
    FUSION_WEIGHT_L4,
    FUSION_WEIGHT_L8,
)
from src.production_inference import ProductionInferenceEngine


# ============================================================
# VISIONFORGE — PRODUCTION LOCALIZATION
# ============================================================
#
# Localization is explanatory only.
#
# Classification remains entirely owned by
# ProductionInferenceEngine.
#
# Production localization:
#   - same canonical production preprocessing
#   - same L4/L8 feature representation
#   - same production reference banks
#   - Euclidean-equivalent patch distance
#   - 50/50 L4 + L8 heatmap fusion
#   - P95 image-level evidence threshold
#   - largest connected component
#
# Classification is NOT changed by this module.
# ============================================================


FEATURE_H = 14
FEATURE_W = 14
L4_CHANNELS = 40
L8_CHANNELS = 48

LOCALIZATION_PERCENTILE = 95.0
MIN_COMPONENT_AREA = 4

LOCALIZATION_METHOD = (
    "l4_l8_patch_distance_p95_largest_component"
)


# ============================================================
# VALIDATION
# ============================================================


def _validate_category(category):
    if category not in CATEGORIES:
        raise ValueError(
            f"Unsupported category: {category}"
        )


def _validate_feature_shapes(l4_queries, l8_queries):
    if l4_queries.ndim != 3:
        raise RuntimeError(
            f"Unexpected L4 query shape: {l4_queries.shape}"
        )

    if l8_queries.ndim != 3:
        raise RuntimeError(
            f"Unexpected L8 query shape: {l8_queries.shape}"
        )

    if l4_queries.shape[1] != FEATURE_H * FEATURE_W:
        raise RuntimeError(
            f"Unexpected L4 patch count: "
            f"{l4_queries.shape}"
        )

    if l8_queries.shape[1] != FEATURE_H * FEATURE_W:
        raise RuntimeError(
            f"Unexpected L8 patch count: "
            f"{l8_queries.shape}"
        )

    if l4_queries.shape[2] != L4_CHANNELS:
        raise RuntimeError(
            f"Unexpected L4 descriptor dimension: "
            f"{l4_queries.shape}"
        )

    if l8_queries.shape[2] != L8_CHANNELS:
        raise RuntimeError(
            f"Unexpected L8 descriptor dimension: "
            f"{l8_queries.shape}"
        )

    if l4_queries.shape[0] != l8_queries.shape[0]:
        raise RuntimeError(
            "L4/L8 batch sizes do not match."
        )


# ============================================================
# PATCH DISTANCES
# ============================================================


def _nearest_patch_distances(
    query_patches,
    reference_bank,
):
    """
    Compute the nearest-reference Euclidean-equivalent
    distance for every query patch.

    Both query and reference descriptors are already L2
    normalized by the production pipeline.

    For normalized vectors:

        d = sqrt(2 - 2 * cosine_similarity)

    The maximum similarity is therefore the nearest reference.
    """

    if query_patches.ndim != 2:
        raise ValueError(
            "query_patches must be 2D."
        )

    if reference_bank.ndim != 2:
        raise ValueError(
            "reference_bank must be 2D."
        )

    if query_patches.shape[1] != reference_bank.shape[1]:
        raise ValueError(
            "Query/reference descriptor dimensions "
            "do not match."
        )

    best_similarity = np.full(
        query_patches.shape[0],
        -np.inf,
        dtype=np.float32,
    )

    num_references = reference_bank.shape[0]

    for start in range(
        0,
        num_references,
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            num_references,
        )

        reference_chunk = reference_bank[
            start:end
        ]

        similarities = (
            query_patches
            @ reference_chunk.T
        )

        chunk_best = np.max(
            similarities,
            axis=1,
        )

        best_similarity = np.maximum(
            best_similarity,
            chunk_best,
        )

    squared_distance = (
        2.0
        - 2.0 * best_similarity
    )

    squared_distance = np.maximum(
        squared_distance,
        0.0,
    )

    distances = np.sqrt(
        squared_distance
    )

    return distances.astype(
        np.float32
    )


# ============================================================
# HEATMAP CONSTRUCTION
# ============================================================


def _reshape_patch_scores(
    patch_scores,
):
    if patch_scores.shape != (
        FEATURE_H * FEATURE_W,
    ):
        raise RuntimeError(
            f"Expected 196 patch scores, "
            f"got {patch_scores.shape}"
        )

    return patch_scores.reshape(
        FEATURE_H,
        FEATURE_W,
    )


def _upsample_heatmap(
    heatmap,
    width,
    height,
):
    return cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    ).astype(
        np.float32
    )


# ============================================================
# EVIDENCE THRESHOLD
# ============================================================


def _p95_threshold(heatmap):
    """
    Image-level P95 threshold used only for localization
    evidence extraction.

    This threshold does NOT affect PASS/REVIEW/FAIL.
    """

    threshold = np.percentile(
        heatmap,
        LOCALIZATION_PERCENTILE,
    )

    return float(threshold)


# ============================================================
# LARGEST CONNECTED COMPONENT
# ============================================================


def _largest_component(
    binary_mask,
):
    """
    Keep only the largest connected evidence component.
    """

    mask_uint8 = (
        binary_mask.astype(
            np.uint8
        )
        * 255
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return np.zeros_like(
            binary_mask,
            dtype=np.uint8,
        )

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_index = (
        1
        + int(
            np.argmax(
                component_areas
            )
        )
    )

    largest_area = int(
        stats[
            largest_index,
            cv2.CC_STAT_AREA,
        ]
    )

    if largest_area < MIN_COMPONENT_AREA:
        return np.zeros_like(
            binary_mask,
            dtype=np.uint8,
        )

    component_mask = (
        labels == largest_index
    ).astype(
        np.uint8
    )

    return component_mask


# ============================================================
# BOUNDING BOX
# ============================================================


def _bounding_box(
    binary_mask,
):
    ys, xs = np.where(
        binary_mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return {
        "x": x1,
        "y": y1,
        "width": x2 - x1 + 1,
        "height": y2 - y1 + 1,
        "x2": x2,
        "y2": y2,
    }


# ============================================================
# CENTER
# ============================================================


def _center(
    bounding_box,
    width,
    height,
):
    if bounding_box is None:
        return None

    center_x = (
        bounding_box["x"]
        + bounding_box["width"] / 2.0
    )

    center_y = (
        bounding_box["y"]
        + bounding_box["height"] / 2.0
    )

    return {
        "x": float(center_x),
        "y": float(center_y),
        "x_normalized": float(
            center_x / max(width, 1)
        ),
        "y_normalized": float(
            center_y / max(height, 1)
        ),
    }


# ============================================================
# HEATMAP STATISTICS
# ============================================================


def _heatmap_statistics(
    heatmap,
):
    return {
        "min": float(
            np.min(heatmap)
        ),
        "max": float(
            np.max(heatmap)
        ),
        "mean": float(
            np.mean(heatmap)
        ),
        "std": float(
            np.std(heatmap)
        ),
        "median": float(
            np.median(heatmap)
        ),
    }


def _region_statistics(
    heatmap,
    mask,
):
    values = heatmap[
        mask > 0
    ]

    if values.size == 0:
        return {
            "pixel_count": 0,
            "mean_anomaly": None,
            "max_anomaly": None,
            "min_anomaly": None,
        }

    return {
        "pixel_count": int(
            values.size
        ),
        "mean_anomaly": float(
            np.mean(values)
        ),
        "max_anomaly": float(
            np.max(values)
        ),
        "min_anomaly": float(
            np.min(values)
        ),
    }


# ============================================================
# HEATMAP NORMALIZATION
# ============================================================


def _normalize_heatmap(
    heatmap,
):
    minimum = np.min(
        heatmap
    )

    maximum = np.max(
        heatmap
    )

    denominator = (
        maximum - minimum
    )

    if denominator <= 1e-12:
        return np.zeros_like(
            heatmap,
            dtype=np.float32,
        )

    normalized = (
        heatmap - minimum
    ) / denominator

    return normalized.astype(
        np.float32
    )


# ============================================================
# SINGLE IMAGE LOCALIZATION
# ============================================================


class ProductionLocalizer:
    """
    Production localization layer built on top of the
    frozen ProductionInferenceEngine.

    It does not modify or replace classification.
    """

    def __init__(
        self,
        engine=None,
    ):
        self.engine = (
            engine
            if engine is not None
            else ProductionInferenceEngine()
        )

    def localize(
        self,
        image_path,
        category,
    ):
        """
        Generate production-consistent localization evidence.

        Returns:
            dict
        """

        _validate_category(
            category
        )

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        with Image.open(
            image_path
        ) as image:
            width, height = image.size

        # --------------------------------------------------------
        # Production feature extraction
        # --------------------------------------------------------

        l4_queries, l8_queries = (
            self.engine.extract_features(
                [image_path]
            )
        )

        _validate_feature_shapes(
            l4_queries,
            l8_queries,
        )

        # --------------------------------------------------------
        # Reference banks
        # --------------------------------------------------------

        l4_bank = (
            self.engine.reference_banks[
                category
            ]["L4"]
            .detach()
            .cpu()
            .numpy()
        )

        l8_bank = (
            self.engine.reference_banks[
                category
            ]["L8"]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------
        # Patch-level anomaly distances
        # --------------------------------------------------------

        l4_patch_scores = (
            _nearest_patch_distances(
                l4_queries[0]
                .detach()
                .cpu()
                .numpy(),
                l4_bank,
            )
        )

        l8_patch_scores = (
            _nearest_patch_distances(
                l8_queries[0]
                .detach()
                .cpu()
                .numpy(),
                l8_bank,
            )
        )

        # --------------------------------------------------------
        # 14 x 14 heatmaps
        # --------------------------------------------------------

        l4_low_res = (
            _reshape_patch_scores(
                l4_patch_scores
            )
        )

        l8_low_res = (
            _reshape_patch_scores(
                l8_patch_scores
            )
        )

        # --------------------------------------------------------
        # L4/L8 production-consistent fusion
        # --------------------------------------------------------

        fused_low_res = (
            FUSION_WEIGHT_L4
            * l4_low_res
            +
            FUSION_WEIGHT_L8
            * l8_low_res
        ).astype(
            np.float32
        )

        # --------------------------------------------------------
        # Original-resolution heatmaps
        # --------------------------------------------------------

        l4_heatmap = _upsample_heatmap(
            l4_low_res,
            width,
            height,
        )

        l8_heatmap = _upsample_heatmap(
            l8_low_res,
            width,
            height,
        )

        fused_heatmap = _upsample_heatmap(
            fused_low_res,
            width,
            height,
        )

        # --------------------------------------------------------
        # P95 evidence threshold
        # --------------------------------------------------------

        evidence_threshold = (
            _p95_threshold(
                fused_heatmap
            )
        )

        binary_mask = (
            fused_heatmap
            >= evidence_threshold
        ).astype(
            np.uint8
        )

        # --------------------------------------------------------
        # Largest connected component
        # --------------------------------------------------------

        evidence_mask = _largest_component(
            binary_mask
        )

        # --------------------------------------------------------
        # Bounding box
        # --------------------------------------------------------

        bounding_box = _bounding_box(
            evidence_mask
        )

        # --------------------------------------------------------
        # Center
        # --------------------------------------------------------

        center = _center(
            bounding_box,
            width,
            height,
        )

        # --------------------------------------------------------
        # Evidence statistics
        # --------------------------------------------------------

        heatmap_stats = (
            _heatmap_statistics(
                fused_heatmap
            )
        )

        region_stats = (
            _region_statistics(
                fused_heatmap,
                evidence_mask,
            )
        )

        # --------------------------------------------------------
        # Visualization heatmap
        # --------------------------------------------------------

        normalized_heatmap = (
            _normalize_heatmap(
                fused_heatmap
            )
        )

        # --------------------------------------------------------
        # Result
        # --------------------------------------------------------

        return {
            "method": LOCALIZATION_METHOD,

            "feature_map": [
                FEATURE_H,
                FEATURE_W,
            ],

            "patch_count": (
                FEATURE_H
                * FEATURE_W
            ),

            "fusion": {
                "l4_weight": float(
                    FUSION_WEIGHT_L4
                ),
                "l8_weight": float(
                    FUSION_WEIGHT_L8
                ),
            },

            "threshold": {
                "method": "P95",
                "value": evidence_threshold,
            },

            "image": {
                "width": int(width),
                "height": int(height),
            },

            "bounding_box": bounding_box,

            "center": center,

            "heatmap_statistics": (
                heatmap_stats
            ),

            "region_statistics": (
                region_stats
            ),

            "heatmap": (
                normalized_heatmap
                .round(4)
                .tolist()
            ),

            "evidence_mask": (
                evidence_mask
                .tolist()
            ),

            "l4_heatmap": (
                _normalize_heatmap(
                    l4_heatmap
                )
                .round(4)
                .tolist()
            ),

            "l8_heatmap": (
                _normalize_heatmap(
                    l8_heatmap
                )
                .round(4)
                .tolist()
            ),
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def localize_image(
    image_path,
    category,
    engine=None,
):
    localizer = ProductionLocalizer(
        engine=engine
    )

    return localizer.localize(
        image_path,
        category,
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    print(
        "Production localization module loaded."
    )