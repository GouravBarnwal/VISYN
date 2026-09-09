import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("data")

BANK_ROOT = Path(
    "artifacts/patch_banks/mobilenet_l8"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# MODEL
# ============================================================

weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

model.eval()

feature_extractor = torch.nn.Sequential(
    *list(model.features[:9])
)

feature_extractor.eval()


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
# SPLITS
# ============================================================

def normal_split(category):

    image_dir = (
        DATA_ROOT
        / category
        / "train"
        / "good"
    )

    images = sorted(
        image_dir.glob("*.png")
    )

    rng = random.Random(SEED)

    images = images.copy()
    rng.shuffle(images)

    n_reference = int(
        0.8 * len(images)
    )

    reference = images[:n_reference]
    development = images[n_reference:]

    return reference, development


def defect_split(category):

    test_dir = (
        DATA_ROOT
        / category
        / "test"
    )

    development = []
    final = []

    defect_types = sorted(
        d.name
        for d in test_dir.iterdir()
        if d.is_dir()
        and d.name != "good"
    )

    # IMPORTANT:
    # This matches the DINOv2 development split exactly.
    rng = random.Random(SEED)

    for defect_type in defect_types:

        images = sorted(
            (
                test_dir
                / defect_type
            ).glob("*.png")
        )

        images = images.copy()
        rng.shuffle(images)

        n_dev = len(images) // 2

        development.extend(
            (
                image,
                defect_type
            )
            for image in images[:n_dev]
        )

        final.extend(
            (
                image,
                defect_type
            )
            for image in images[n_dev:]
        )

    return development, final

# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_patches(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0)

    features = feature_extractor(
        tensor
    )

    # Expected:
    # [1, 48, 14, 14]

    features = features[0]

    # [48, 14, 14]
    features = features.permute(
        1, 2, 0
    )

    # [196, 48]
    patches = features.reshape(
        -1,
        features.shape[-1]
    )

    patches = F.normalize(
        patches,
        p=2,
        dim=1
    )

    return patches


# ============================================================
# NEAREST NEIGHBOR
# ============================================================

@torch.no_grad()
def nearest_distances(
    query,
    bank
):

    # Both query and bank are already
    # L2-normalized.

    distances = torch.cdist(
        query,
        bank,
        p=2
    )

    return distances.min(
        dim=1
    ).values


# ============================================================
# AGGREGATION
# ============================================================

def aggregate(distances):

    values = (
        distances
        .cpu()
        .numpy()
    )

    values = np.sort(values)

    return {
        "max": float(
            values[-1]
        ),

        "top5": float(
            np.mean(
                values[-5:]
            )
        ),

        "top10": float(
            np.mean(
                values[-10:]
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

results = {}

for category in CATEGORIES:

    print()
    print("=" * 60)
    print(category)
    print("=" * 60)

    bank_path = (
        BANK_ROOT
        / f"{category}.npy"
    )

    bank_np = np.load(
        bank_path
    ).astype(
        np.float32
    )

    bank = torch.from_numpy(
        bank_np
    )

    # Verify bank normalization
    norms = torch.linalg.norm(
        bank,
        dim=1
    )

    print(
        "Bank:",
        bank.shape
    )

    print(
        "Bank norm:",
        f"{norms.mean():.6f}"
    )

    _, normal_dev = normal_split(
        category
    )

    defect_dev, _ = defect_split(
        category
    )

    print(
        "Normal dev:",
        len(normal_dev)
    )

    print(
        "Defect dev:",
        len(defect_dev)
    )

    category_results = {
        "normal_dev": [],
        "defect_dev": [],
    }


    # --------------------------------------------------------
    # NORMAL DEVELOPMENT
    # --------------------------------------------------------

    for i, image_path in enumerate(
        normal_dev
    ):

        start = time.perf_counter()

        patches = extract_patches(
            image_path
        )

        distances = nearest_distances(
            patches,
            bank
        )

        scores = aggregate(
            distances
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        category_results[
            "normal_dev"
        ].append({
            "path": str(
                image_path.relative_to(
                    DATA_ROOT
                )
            ),
            **scores,
            "inference_ms": float(
                elapsed_ms
            ),
        })

        if (
            (i + 1) % 10 == 0
            or i + 1 == len(normal_dev)
        ):
            print(
                f"Normal "
                f"{i + 1}/"
                f"{len(normal_dev)}"
            )


    # --------------------------------------------------------
    # DEFECT DEVELOPMENT
    # --------------------------------------------------------

    for i, (
        image_path,
        defect_type
    ) in enumerate(defect_dev):

        start = time.perf_counter()

        patches = extract_patches(
            image_path
        )

        distances = nearest_distances(
            patches,
            bank
        )

        scores = aggregate(
            distances
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        category_results[
            "defect_dev"
        ].append({
            "path": str(
                image_path.relative_to(
                    DATA_ROOT
                )
            ),
            "defect_type": defect_type,
            **scores,
            "inference_ms": float(
                elapsed_ms
            ),
        })

        if (
            (i + 1) % 10 == 0
            or i + 1 == len(defect_dev)
        ):
            print(
                f"Defect "
                f"{i + 1}/"
                f"{len(defect_dev)}"
            )


    results[category] = (
        category_results
    )

    print(
        f"Finished {category}"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print("=" * 60)
print(
    "MOBILENET L8 DEVELOPMENT SCORING COMPLETE"
)
print("=" * 60)

print(
    "Saved:",
    OUTPUT_PATH
)