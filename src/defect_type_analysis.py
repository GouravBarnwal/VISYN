from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


DATA_ROOT = Path("data")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

SEED = 42
REFERENCE_RATIO = 0.8


# ============================================================
# Model
# ============================================================

weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT

model = torchvision.models.mobilenet_v3_small(
    weights=weights
)

feature_extractor = torch.nn.Sequential(
    model.features[:9]
)

feature_extractor.eval()

transform = weights.transforms()


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    features = feature_extractor(tensor)

    features = features.squeeze(0)

    patches = features.permute(
        1, 2, 0
    ).reshape(
        -1,
        features.shape[0]
    )

    return patches.numpy()


# ============================================================
# Reference images
# ============================================================

def get_reference_images(category):

    train_dir = (
        DATA_ROOT
        / category
        / "train"
        / "good"
    )

    images = sorted(
        train_dir.glob("*.png")
    )

    rng = np.random.default_rng(SEED)

    indices = np.arange(len(images))
    rng.shuffle(indices)

    split = int(
        len(indices) * REFERENCE_RATIO
    )

    return [
        images[i]
        for i in indices[:split]
    ]


# ============================================================
# Test images grouped by defect type
# ============================================================

def get_test_groups(category):

    test_dir = DATA_ROOT / category / "test"

    groups = {}

    for defect_dir in sorted(
        test_dir.iterdir()
    ):

        if not defect_dir.is_dir():
            continue

        if defect_dir.name == "good":
            continue

        images = sorted(
            defect_dir.glob("*.png")
        )

        groups[defect_dir.name] = images

    return groups


# ============================================================
# TOP-5 anomaly score
# ============================================================

def calculate_score(image_path, nn):

    patches = extract_features(
        image_path
    )

    distances, _ = nn.kneighbors(
        patches,
        return_distance=True
    )

    distances = distances.ravel()

    return np.mean(
        np.sort(distances)[-5:]
    )


# ============================================================
# Analyze category
# ============================================================

def analyze_category(category):

    print("\n" + "=" * 70)
    print(f"DEFECT-TYPE ANALYSIS: {category}")
    print("=" * 70)

    reference_images = get_reference_images(
        category
    )

    defect_groups = get_test_groups(
        category
    )

    # --------------------------------------------------------
    # Build reference bank
    # --------------------------------------------------------

    patches = []

    for i, image_path in enumerate(
        reference_images
    ):

        patches.append(
            extract_features(image_path)
        )

        if (i + 1) % 50 == 0:
            print(
                f"Reference features: "
                f"{i + 1}/{len(reference_images)}"
            )

    patch_bank = np.concatenate(
        patches,
        axis=0
    )

    print(
        "Patch bank:",
        patch_bank.shape
    )

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    # --------------------------------------------------------
    # Score normal test images
    # --------------------------------------------------------

    test_dir = DATA_ROOT / category / "test"
    good_dir = test_dir / "good"

    good_images = sorted(
        good_dir.glob("*.png")
    )

    good_scores = np.array([
        calculate_score(
            path,
            nn
        )
        for path in good_images
    ])

    # --------------------------------------------------------
    # Analyze each defect type
    # --------------------------------------------------------

    print("\nPer-defect-type results")
    print("-" * 70)

    print(
        f"{'Defect type':<25}"
        f"{'N':>6}"
        f"{'AUROC':>10}"
        f"{'Median':>12}"
        f"{'Min':>12}"
    )

    print("-" * 70)

    category_results = []

    for defect_type, images in defect_groups.items():

        defect_scores = np.array([
            calculate_score(
                path,
                nn
            )
            for path in images
        ])

        labels = np.concatenate([
            np.zeros(len(good_scores)),
            np.ones(len(defect_scores))
        ])

        scores = np.concatenate([
            good_scores,
            defect_scores
        ])

        auc = roc_auc_score(
            labels,
            scores
        )

        result = {
            "category": category,
            "defect_type": defect_type,
            "count": len(images),
            "auroc": auc,
            "median": np.median(defect_scores),
            "minimum": np.min(defect_scores),
        }

        category_results.append(result)

        print(
            f"{defect_type:<25}"
            f"{len(images):>6}"
            f"{auc:>10.4f}"
            f"{np.median(defect_scores):>12.4f}"
            f"{np.min(defect_scores):>12.4f}"
        )

    return category_results


# ============================================================
# Main
# ============================================================

all_results = []

for category in CATEGORIES:

    results = analyze_category(
        category
    )

    all_results.extend(
        results
    )


# ============================================================
# Overall summary
# ============================================================

print("\n\n" + "=" * 70)
print("ALL DEFECT TYPES — SUMMARY")
print("=" * 70)

print(
    f"{'Category':<15}"
    f"{'Defect type':<25}"
    f"{'N':>6}"
    f"{'AUROC':>10}"
)

print("-" * 60)

for result in all_results:

    print(
        f"{result['category']:<15}"
        f"{result['defect_type']:<25}"
        f"{result['count']:>6}"
        f"{result['auroc']:>10.4f}"
    )

print("\nAnalysis complete.")