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

OUTPUT_DIR = Path(
    "artifacts/evaluation/localization/"
    "visualizations_hybrid"
)

CATEGORY_COUNT = 3

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

    # Normalize each spatial descriptor.
    norms = np.linalg.norm(
        feature_map,
        axis=2,
        keepdims=True,
    )

    feature_map = (
        feature_map
        / np.maximum(norms, 1e-12)
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

    similarity = query @ global_bank.T

    nearest_similarity = np.max(
        similarity,
        axis=1,
    )

    anomaly = (
        1.0 - nearest_similarity
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
# MASK
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
# VISUALIZATION
# ============================================================

def save_visualization(
    image_path,
    mask_path,
    heatmap,
    output_path,
):

    image = cv2.imread(
        str(
            resolve_image_path(
                image_path
            )
        )
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    mask = load_mask(
        mask_path
    )

    if mask.shape != (
        height,
        width,
    ):

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    # Visualization-only normalization.
    heatmap_norm = (
        heatmap
        - heatmap.min()
    )

    denominator = (
        heatmap_norm.max()
        + 1e-12
    )

    heatmap_norm = (
        heatmap_norm
        / denominator
    )

    heatmap_uint8 = (
        heatmap_norm * 255
    ).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image,
        0.60,
        heatmap_color,
        0.40,
        0,
    )

    mask_visual = (
        mask * 255
    ).astype(np.uint8)

    mask_visual = cv2.cvtColor(
        mask_visual,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.concatenate(
        [
            image,
            mask_visual,
            heatmap_color,
            overlay,
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — HYBRID LOCALIZATION VISUALIZATION"
    )
    print("=" * 72)

    print(
        f"Alpha: {ALPHA:.2f} "
        f"(25% global + 75% spatial)"
    )

    print(
        "Evaluation source: development defect split"
    )

    print(
        "Final test used: False"
    )

    print()

    # --------------------------------------------------------
    # LOAD DEFECT SPLIT
    # --------------------------------------------------------

    with open(
        DEFECT_SPLIT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        defect_splits = json.load(f)

    # IMPORTANT:
    # defect_splits.json contains metadata at the top level.
    # The actual categories live under "categories".
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

    categories = list(
        categories_data.keys()
    )

    # --------------------------------------------------------
    # CATEGORY LOOP
    # --------------------------------------------------------

    for category in categories:

        print("-" * 72)
        print(
            category.upper()
        )
        print("-" * 72)

        # ----------------------------------------------------
        # BANKS
        # ----------------------------------------------------

        global_bank_path = (
            GLOBAL_BANK_DIR
            / f"{category}.npy"
        )

        spatial_bank_path = (
            SPATIAL_BANK_DIR
            / f"{category}.npy"
        )

        if not global_bank_path.exists():

            print(
                f"Missing global bank: "
                f"{global_bank_path}"
            )

            continue

        if not spatial_bank_path.exists():

            print(
                f"Missing spatial bank: "
                f"{spatial_bank_path}"
            )

            continue

        global_bank = np.load(
            global_bank_path
        )

        spatial_bank = np.load(
            spatial_bank_path
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
        # DEVELOPMENT DEFECTS
        # ----------------------------------------------------

        development = categories_data[
            category
        ]["development"]

        print(
            f"Development defects: "
            f"{len(development)}"
        )

        # ----------------------------------------------------
        # SELECT REPRESENTATIVE IMAGES
        # ----------------------------------------------------

        selected = sorted(
            development
        )[:CATEGORY_COUNT]

        print(
            f"Visualizing "
            f"{len(selected)} images..."
        )

        for index, image_path in enumerate(
            selected,
            start=1,
        ):

            mask_path = find_mask(
                image_path
            )

            if mask_path is None:

                print(
                    f"Skipping {image_path}: "
                    "mask not found"
                )

                continue

            print(
                f"[{index}/{len(selected)}] "
                f"{image_path}"
            )

            # ------------------------------------------------
            # FEATURE EXTRACTION
            # ------------------------------------------------

            feature_map = (
                extract_feature_map(
                    model,
                    image_path,
                )
            )

            # ------------------------------------------------
            # GLOBAL
            # ------------------------------------------------

            global_map = (
                global_heatmap(
                    feature_map,
                    global_bank,
                )
            )

            # ------------------------------------------------
            # SPATIAL
            # ------------------------------------------------

            spatial_map = (
                spatial_heatmap(
                    feature_map,
                    spatial_bank,
                )
            )

            # ------------------------------------------------
            # HYBRID
            # ------------------------------------------------

            hybrid_map = (
                hybrid_heatmap(
                    global_map,
                    spatial_map,
                    ALPHA,
                )
            )

            # ------------------------------------------------
            # ORIGINAL IMAGE SIZE
            # ------------------------------------------------

            image = cv2.imread(
                str(
                    resolve_image_path(
                        image_path
                    )
                )
            )

            height, width = (
                image.shape[:2]
            )

            hybrid_full = (
                upsample_heatmap(
                    hybrid_map,
                    (width, height),
                )
            )

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            stem = Path(
                image_path
            ).stem

            output_path = (
                OUTPUT_DIR
                / category
                / (
                    f"{stem}"
                    f"_hybrid_alpha_{ALPHA:.2f}.png"
                )
            )

            save_visualization(
                image_path,
                mask_path,
                hybrid_full,
                output_path,
            )

        print()

    print("=" * 72)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Saved under: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()