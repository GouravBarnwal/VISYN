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
# Image split
# ============================================================

def get_reference_images(category):

    train_dir = (
        DATA_ROOT
        / category
        / "train"
        / "good"
    )

    images = sorted(train_dir.glob("*.png"))

    rng = np.random.default_rng(SEED)

    indices = np.arange(len(images))
    rng.shuffle(indices)

    split = int(len(indices) * REFERENCE_RATIO)

    return [
        images[i]
        for i in indices[:split]
    ]


# ============================================================
# Test images
# ============================================================

def get_test_images(category):

    test_dir = DATA_ROOT / category / "test"

    good = []
    defective = []

    for defect_dir in sorted(test_dir.iterdir()):

        if not defect_dir.is_dir():
            continue

        images = sorted(
            defect_dir.glob("*.png")
        )

        if defect_dir.name == "good":
            good.extend(images)
        else:
            defective.extend(images)

    return good, defective


# ============================================================
# TOP-5 anomaly score
# ============================================================

def calculate_score(image_path, nn):

    patches = extract_features(image_path)

    distances, _ = nn.kneighbors(
        patches,
        return_distance=True
    )

    distances = distances.ravel()

    return np.mean(
        np.sort(distances)[-5:]
    )


# ============================================================
# Evaluate category
# ============================================================

def evaluate_category(category):

    print("\n" + "=" * 70)
    print(f"SCORE DIAGNOSIS: {category}")
    print("=" * 70)

    reference_images = get_reference_images(
        category
    )

    good_images, defective_images = (
        get_test_images(category)
    )

    print(
        f"Reference: {len(reference_images)}"
    )

    print(
        f"Test good: {len(good_images)}"
    )

    print(
        f"Test defective: {len(defective_images)}"
    )

    # --------------------------------------------------------
    # Build patch bank
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
    # Score test images
    # --------------------------------------------------------

    good_scores = []

    for image_path in good_images:

        score = calculate_score(
            image_path,
            nn
        )

        good_scores.append(score)

    defective_scores = []

    for image_path in defective_images:

        score = calculate_score(
            image_path,
            nn
        )

        defective_scores.append(score)

    good_scores = np.asarray(
        good_scores
    )

    defective_scores = np.asarray(
        defective_scores
    )

    scores = np.concatenate([
        good_scores,
        defective_scores
    ])

    labels = np.concatenate([
        np.zeros(len(good_scores)),
        np.ones(len(defective_scores))
    ])

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    auc = roc_auc_score(
        labels,
        scores
    )

    # --------------------------------------------------------
    # Distribution statistics
    # --------------------------------------------------------

    print("\nContinuous-score performance")
    print("-" * 50)

    print(
        f"AUROC: {auc:.4f}"
    )

    print("\nGOOD")
    print(
        f"Median: {np.median(good_scores):.6f}"
    )

    print(
        f"P95:    {np.percentile(good_scores, 95):.6f}"
    )

    print(
        f"Maximum:{np.max(good_scores):.6f}"
    )

    print("\nDEFECTIVE")
    print(
        f"Median: {np.median(defective_scores):.6f}"
    )

    print(
        f"P25:    {np.percentile(defective_scores, 25):.6f}"
    )

    print(
        f"P75:    {np.percentile(defective_scores, 75):.6f}"
    )

    print(
        f"Minimum:{np.min(defective_scores):.6f}"
    )

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    good_median = np.median(good_scores)
    defect_median = np.median(defective_scores)

    print("\nInterpretation")
    print("-" * 50)

    if auc >= 0.95:
        print(
            "Strong score separation."
        )

    elif auc >= 0.85:
        print(
            "Moderate/good score separation."
        )

    elif auc >= 0.70:
        print(
            "Weak score separation."
        )

    else:
        print(
            "Poor score separation."
        )

    if defect_median > good_median:
        print(
            "Defect median is above good median."
        )
    else:
        print(
            "Defect median is NOT above good median."
        )

    return {
        "category": category,
        "auroc": auc,
        "good_median": good_median,
        "good_max": np.max(good_scores),
        "defect_median": defect_median,
        "defect_min": np.min(defective_scores),
    }


# ============================================================
# Main
# ============================================================

results = []

for category in CATEGORIES:

    results.append(
        evaluate_category(category)
    )


# ============================================================
# Summary
# ============================================================

print("\n\n" + "=" * 70)
print("CONTINUOUS SCORE DIAGNOSIS SUMMARY")
print("=" * 70)

print(
    f"{'Category':<15}"
    f"{'AUROC':>10}"
    f"{'Good Med':>12}"
    f"{'Defect Med':>13}"
)

print("-" * 55)

for r in results:

    print(
        f"{r['category']:<15}"
        f"{r['auroc']:>10.4f}"
        f"{r['good_median']:>12.4f}"
        f"{r['defect_median']:>13.4f}"
    )

print("\nDiagnosis complete.")