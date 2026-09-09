from pathlib import Path
import sys

import numpy as np
from PIL import Image


# ============================================================
# IMPORT VISYN LOCALIZATION EVIDENCE MODULE
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from localization_evidence import (
    create_inspection_evidence,
)


# ============================================================
# TEST CONFIG
# ============================================================

IMAGE_PATH = Path(
    "data/bottle/test/broken_large/004.png"
)

HEATMAP_SIZE = (
    14,
    14,
)

TOP_PERCENT = 0.05


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — LOCALIZATION EVIDENCE TEST"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    )

    width, height = image.size

    print(
        f"Test image: {IMAGE_PATH}"
    )

    print(
        f"Image size: {width} x {height}"
    )

    # --------------------------------------------------------
    # Create a synthetic anomaly heatmap.
    #
    # This test intentionally does NOT run MobileNet.
    # It only verifies that the evidence-processing module
    # correctly handles a heatmap.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    heatmap = rng.random(
        HEATMAP_SIZE,
        dtype=np.float32,
    )

    # Create a strong artificial anomaly region
    # so the bounding-box extraction has something
    # obvious to find.
    heatmap[5:8, 6:9] += 2.0

    print(
        f"Input heatmap: "
        f"{heatmap.shape}"
    )

    print(
        f"Top anomaly percentage: "
        f"{TOP_PERCENT * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Generate evidence
    # --------------------------------------------------------

    evidence = create_inspection_evidence(
        heatmap=heatmap,
        image_width=width,
        image_height=height,
        top_percent=TOP_PERCENT,
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_keys = [
        "method",
        "top_percent",
        "heatmap",
        "evidence_mask",
        "bounding_box",
        "center",
        "heatmap_statistics",
        "region_statistics",
    ]

    for key in required_keys:

        if key not in evidence:

            raise RuntimeError(
                f"Missing evidence key: {key}"
            )

    # --------------------------------------------------------
    # Validate heatmap dimensions
    # --------------------------------------------------------

    output_heatmap = np.asarray(
        evidence["heatmap"]
    )

    expected_shape = (
        height,
        width,
    )

    if output_heatmap.shape != expected_shape:

        raise RuntimeError(
            "Heatmap shape mismatch: "
            f"expected {expected_shape}, "
            f"got {output_heatmap.shape}"
        )

    # --------------------------------------------------------
    # Validate evidence mask dimensions
    # --------------------------------------------------------

    output_mask = np.asarray(
        evidence["evidence_mask"]
    )

    if output_mask.shape != expected_shape:

        raise RuntimeError(
            "Evidence mask shape mismatch: "
            f"expected {expected_shape}, "
            f"got {output_mask.shape}"
        )

    # --------------------------------------------------------
    # Validate mask values
    # --------------------------------------------------------

    unique_values = np.unique(
        output_mask
    )

    if not np.all(
        np.isin(
            unique_values,
            [0, 1],
        )
    ):

        raise RuntimeError(
            "Evidence mask contains values "
            "other than 0 and 1."
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("-" * 72)
    print(
        "EVIDENCE STRUCTURE"
    )
    print("-" * 72)

    print(
        f"Method: "
        f"{evidence['method']}"
    )

    print(
        f"Top percentage: "
        f"{evidence['top_percent']}"
    )

    print(
        f"Bounding box: "
        f"{evidence['bounding_box']}"
    )

    print(
        f"Center: "
        f"{evidence['center']}"
    )

    print()

    print(
        "Heatmap statistics:"
    )

    for key, value in (
        evidence[
            "heatmap_statistics"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        "Top-region statistics:"
    )

    for key, value in (
        evidence[
            "region_statistics"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    bounding_box = evidence[
        "bounding_box"
    ]

    if bounding_box is None:

        raise RuntimeError(
            "Expected a bounding box, "
            "but received None."
        )

    if not (
        0
        <= bounding_box["x"]
        < width
    ):

        raise RuntimeError(
            "Bounding-box x coordinate "
            "is outside image."
        )

    if not (
        0
        <= bounding_box["y"]
        < height
    ):

        raise RuntimeError(
            "Bounding-box y coordinate "
            "is outside image."
        )

    print()
    print("=" * 72)
    print(
        "LOCALIZATION EVIDENCE TEST PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()