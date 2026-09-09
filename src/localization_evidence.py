from pathlib import Path

import cv2
import numpy as np


# ============================================================
# VISYN — LOCALIZATION EVIDENCE
# ============================================================
#
# Converts an anomaly heatmap into structured inspection
# evidence that can later be consumed by the API/UI.
#
# Current strategy:
#   - Use the top 5% highest anomaly pixels
#   - Build a binary evidence mask
#   - Extract the bounding box
#   - Report useful localization statistics
#
# This module does NOT make PASS/REVIEW/FAIL decisions.
# Calibration and decision policy are handled separately.
# ============================================================


DEFAULT_TOP_PERCENT = 0.05


# ============================================================
# TOP ANOMALY MASK
# ============================================================

def create_top_percent_mask(
    heatmap,
    percentage=DEFAULT_TOP_PERCENT,
):
    """
    Convert an anomaly heatmap into a binary mask containing
    the highest-scoring percentage of pixels.

    Parameters
    ----------
    heatmap : np.ndarray
        2D anomaly heatmap.

    percentage : float
        Fraction of pixels to retain.

        Example:
            0.05 = top 5%

    Returns
    -------
    np.ndarray
        Binary uint8 mask with values 0 or 1.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be a 2D array."
        )

    if not (
        0.0 < percentage <= 1.0
    ):
        raise ValueError(
            "percentage must be in (0, 1]."
        )

    flat = heatmap.reshape(-1)

    total_pixels = flat.size

    if total_pixels == 0:
        raise ValueError(
            "heatmap contains no pixels."
        )

    number_of_pixels = max(
        1,
        int(
            np.ceil(
                total_pixels
                * percentage
            )
        ),
    )

    threshold_index = (
        total_pixels
        - number_of_pixels
    )

    threshold = np.partition(
        flat,
        threshold_index,
    )[threshold_index]

    mask = (
        heatmap >= threshold
    ).astype(np.uint8)

    return mask


# ============================================================
# BOUNDING BOX
# ============================================================

def extract_bounding_box(
    binary_mask,
):
    """
    Extract the smallest axis-aligned bounding box
    containing all positive pixels.

    Returns
    -------
    dict or None

    Example
    -------
    {
        "x": 120,
        "y": 80,
        "width": 140,
        "height": 100,
        "x2": 259,
        "y2": 179
    }
    """

    mask = np.asarray(
        binary_mask
    )

    if mask.ndim != 2:
        raise ValueError(
            "binary_mask must be 2D."
        )

    ys, xs = np.where(
        mask > 0
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
# ANOMALY STATISTICS
# ============================================================

def calculate_heatmap_statistics(
    heatmap,
):
    """
    Calculate basic statistics describing the anomaly map.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be 2D."
        )

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


# ============================================================
# TOP REGION STATISTICS
# ============================================================

def calculate_top_region_statistics(
    heatmap,
    evidence_mask,
):
    """
    Calculate statistics specifically over the selected
    anomalous evidence region.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    evidence_mask = np.asarray(
        evidence_mask
    )

    values = heatmap[
        evidence_mask > 0
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
# NORMALIZED CENTER
# ============================================================

def calculate_region_center(
    bounding_box,
    image_width,
    image_height,
):
    """
    Calculate the center of the evidence bounding box.

    Also returns normalized coordinates so the representation
    is independent of image resolution.
    """

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
            center_x / max(
                image_width,
                1,
            )
        ),
        "y_normalized": float(
            center_y / max(
                image_height,
                1,
            )
        ),
    }


# ============================================================
# HEATMAP NORMALIZATION
# ============================================================

def normalize_heatmap(
    heatmap,
):
    """
    Normalize an anomaly heatmap to [0, 1].

    This is intended for visualization/API output.

    It does NOT alter the underlying anomaly score.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    minimum = np.min(
        heatmap
    )

    maximum = np.max(
        heatmap
    )

    denominator = (
        maximum
        - minimum
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
# RESIZE HEATMAP
# ============================================================

def resize_heatmap(
    heatmap,
    width,
    height,
):
    """
    Resize a low-resolution anomaly heatmap to image
    resolution.

    Bicubic interpolation is used for visualization.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be 2D."
        )

    resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    return resized.astype(
        np.float32
    )


# ============================================================
# CREATE INSPECTION EVIDENCE
# ============================================================

def create_inspection_evidence(
    heatmap,
    image_width,
    image_height,
    top_percent=DEFAULT_TOP_PERCENT,
):
    """
    Convert an anomaly heatmap into a structured evidence
    object.

    Parameters
    ----------
    heatmap : np.ndarray
        2D anomaly heatmap.

    image_width : int
        Width of the original image.

    image_height : int
        Height of the original image.

    top_percent : float
        Percentage of highest-scoring pixels used as the
        evidence region.

    Returns
    -------
    dict
        Structured localization evidence.
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Resize to original image dimensions if necessary.
    # --------------------------------------------------------

    if heatmap.shape != (
        image_height,
        image_width,
    ):

        heatmap_full = resize_heatmap(
            heatmap,
            image_width,
            image_height,
        )

    else:

        heatmap_full = heatmap

    # --------------------------------------------------------
    # Evidence mask
    # --------------------------------------------------------

    evidence_mask = (
        create_top_percent_mask(
            heatmap_full,
            top_percent,
        )
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bounding_box = (
        extract_bounding_box(
            evidence_mask
        )
    )

    # --------------------------------------------------------
    # Region center
    # --------------------------------------------------------

    center = calculate_region_center(
        bounding_box,
        image_width,
        image_height,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    heatmap_statistics = (
        calculate_heatmap_statistics(
            heatmap_full
        )
    )

    region_statistics = (
        calculate_top_region_statistics(
            heatmap_full,
            evidence_mask,
        )
    )

    # --------------------------------------------------------
    # Normalized heatmap
    # --------------------------------------------------------

    normalized_heatmap = (
        normalize_heatmap(
            heatmap_full
        )
    )

    # --------------------------------------------------------
    # Convert heatmap to compact Python lists.
    #
    # This is convenient for JSON/API experiments.
    # Production API may later use a compressed encoding.
    # --------------------------------------------------------

    heatmap_output = (
        normalized_heatmap
        .round(4)
        .tolist()
    )

    # --------------------------------------------------------
    # Evidence mask
    # --------------------------------------------------------

    mask_output = (
        evidence_mask
        .tolist()
    )

    # --------------------------------------------------------
    # Final evidence object
    # --------------------------------------------------------

    evidence = {
        "method": "top_percent_anomaly_region",

        "top_percent": float(
            top_percent
        ),

        "heatmap": heatmap_output,

        "evidence_mask": mask_output,

        "bounding_box": bounding_box,

        "center": center,

        "heatmap_statistics":
            heatmap_statistics,

        "region_statistics":
            region_statistics,
    }

    return evidence