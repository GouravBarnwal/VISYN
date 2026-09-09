from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# IMPORT LOCALIZATION EVIDENCE MODULE
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from localization_evidence import (
    create_inspection_evidence,
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

IMAGE_PATH = Path(
    "data/bottle/test/broken_large/004.png"
)

GLOBAL_BANK_PATH = Path(
    "artifacts/patch_banks/"
    "mobilenet_l8/bottle.npy"
)

SPATIAL_BANK_PATH = Path(
    "artifacts/patch_banks/"
    "mobilenet_l8_spatial/bottle.npy"
)

ALPHA = 0.75

TOP_PERCENT = 0.05


# ============================================================
# MODEL
# ============================================================

class MobileNetL8(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=(
                models.MobileNet_V3_Small_Weights.DEFAULT
            )
        )

        self.features = backbone.features[:9]

    def forward(self, x):

        return self.features(x)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],

        std=[
            0.229,
            0.224,
            0.225,
        ],
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

    feature_map = model(
        x
    )

    feature_map = (
        feature_map
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    # L2 normalize every spatial descriptor.
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
        query
        @ global_bank.T
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

            query = (
                feature_map[r, c]
            )

            references = (
                spatial_bank[
                    :,
                    r,
                    c,
                    :
                ]
            )

            similarities = (
                references
                @ query
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
        (1.0 - alpha)
        * global_map
        +
        alpha
        * spatial_map
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISYN — REAL LOCALIZATION "
        "EVIDENCE INTEGRATION TEST"
    )
    print("=" * 72)

    print(
        f"Image: {IMAGE_PATH}"
    )

    print(
        "Representation: "
        "MobileNetV3-Small L8"
    )

    print(
        "Localization: "
        "hybrid global + spatial"
    )

    print(
        f"Alpha: {ALPHA}"
    )

    print(
        f"Evidence region: "
        f"top {TOP_PERCENT * 100:.1f}%"
    )

    print()

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        IMAGE_PATH,
        GLOBAL_BANK_PATH,
        SPATIAL_BANK_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # LOAD BANKS
    # --------------------------------------------------------

    global_bank = np.load(
        GLOBAL_BANK_PATH
    )

    spatial_bank = np.load(
        SPATIAL_BANK_PATH
    )

    print(
        f"Global bank: "
        f"{global_bank.shape}"
    )

    print(
        f"Spatial bank: "
        f"{spatial_bank.shape}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MobileNetL8()
    model.eval()

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    feature_map = (
        extract_feature_map(
            model,
            IMAGE_PATH,
        )
    )

    print(
        f"Feature map: "
        f"{feature_map.shape}"
    )

    expected_shape = (
        14,
        14,
        48,
    )

    if feature_map.shape != expected_shape:

        raise RuntimeError(
            "Unexpected feature-map shape: "
            f"{feature_map.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # GLOBAL LOCALIZATION
    # --------------------------------------------------------

    global_map = (
        global_heatmap(
            feature_map,
            global_bank,
        )
    )

    print(
        f"Global heatmap: "
        f"{global_map.shape}"
    )

    # --------------------------------------------------------
    # SPATIAL LOCALIZATION
    # --------------------------------------------------------

    spatial_map = (
        spatial_heatmap(
            feature_map,
            spatial_bank,
        )
    )

    print(
        f"Spatial heatmap: "
        f"{spatial_map.shape}"
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    hybrid_map = (
        hybrid_heatmap(
            global_map,
            spatial_map,
            ALPHA,
        )
    )

    print(
        f"Hybrid heatmap: "
        f"{hybrid_map.shape}"
    )

    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    )

    width, height = (
        image.size
    )

    print(
        f"Original image: "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # CREATE EVIDENCE
    # --------------------------------------------------------

    evidence = (
        create_inspection_evidence(
            heatmap=hybrid_map,
            image_width=width,
            image_height=height,
            top_percent=TOP_PERCENT,
        )
    )

    # --------------------------------------------------------
    # VALIDATE
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
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print("-" * 72)
    print(
        "REAL EVIDENCE RESULT"
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
            f"  {key}: {value:.6f}"
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
    # VALIDATE HEATMAP
    # --------------------------------------------------------

    heatmap_output = np.asarray(
        evidence["heatmap"]
    )

    if heatmap_output.shape != (
        height,
        width,
    ):

        raise RuntimeError(
            "Evidence heatmap shape mismatch: "
            f"{heatmap_output.shape}"
        )

    # --------------------------------------------------------
    # VALIDATE MASK
    # --------------------------------------------------------

    evidence_mask = np.asarray(
        evidence["evidence_mask"]
    )

    if evidence_mask.shape != (
        height,
        width,
    ):

        raise RuntimeError(
            "Evidence mask shape mismatch: "
            f"{evidence_mask.shape}"
        )

    # --------------------------------------------------------
    # VALIDATE BOUNDING BOX
    # --------------------------------------------------------

    bounding_box = evidence[
        "bounding_box"
    ]

    if bounding_box is None:

        raise RuntimeError(
            "Real anomaly heatmap produced "
            "no evidence bounding box."
        )

    if bounding_box["x"] < 0:

        raise RuntimeError(
            "Bounding box x < 0."
        )

    if bounding_box["y"] < 0:

        raise RuntimeError(
            "Bounding box y < 0."
        )

    if (
        bounding_box["x2"]
        >= width
    ):

        raise RuntimeError(
            "Bounding box exceeds image width."
        )

    if (
        bounding_box["y2"]
        >= height
    ):

        raise RuntimeError(
            "Bounding box exceeds image height."
        )

    # --------------------------------------------------------
    # CHECK NORMALIZED CENTER
    # --------------------------------------------------------

    center = evidence[
        "center"
    ]

    if center is None:

        raise RuntimeError(
            "Evidence center is missing."
        )

    if not (
        0.0
        <= center["x_normalized"]
        <= 1.0
    ):

        raise RuntimeError(
            "Normalized x center outside [0, 1]."
        )

    if not (
        0.0
        <= center["y_normalized"]
        <= 1.0
    ):

        raise RuntimeError(
            "Normalized y center outside [0, 1]."
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "REAL LOCALIZATION EVIDENCE "
        "INTEGRATION TEST PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()