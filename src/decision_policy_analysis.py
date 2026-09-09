from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score


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
# TOP-5 score
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
# Analyze one category
# ============================================================

def analyze_category(category):

    print("\n" + "=" * 70)
    print(f"DECISION POLICY ANALYSIS: {category}")
    print("=" * 70)

    reference_images = get_reference_images(category)

    good_images, defective_images = get_test_images(
        category
    )

    # --------------------------------------------------------
    # Build patch bank
    # --------------------------------------------------------

    patches = []

    for i, image_path in enumerate(reference_images):

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

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    # --------------------------------------------------------
    # Score test images
    # --------------------------------------------------------

    good_scores = np.array([
        calculate_score(path, nn)
        for path in good_images
    ])

    defect_scores = np.array([
        calculate_score(path, nn)
        for path in defective_images
    ])

    scores = np.concatenate([
        good_scores,
        defect_scores
    ])

    labels = np.concatenate([
        np.zeros(len(good_scores)),
        np.ones(len(defect_scores))
    ])

    # --------------------------------------------------------
    # Candidate thresholds
    # --------------------------------------------------------

    # Use percentiles of the combined score distribution
    # ONLY for diagnostic analysis.
    #
    # These values are NOT final production thresholds.

    thresholds = np.percentile(
        scores,
        np.arange(5, 96, 5)
    )

    print("\nThreshold sweep")
    print("-" * 70)

    print(
        f"{'Threshold':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'FP':>8}"
        f"{'FN':>8}"
    )

    best_f1 = -1
    best_threshold = None

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        fp = np.sum(
            (labels == 0)
            &
            (predictions == 1)
        )

        fn = np.sum(
            (labels == 1)
            &
            (predictions == 0)
        )

        print(
            f"{threshold:>12.4f}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{f1:>12.4f}"
            f"{fp:>8}"
            f"{fn:>8}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("\nDiagnostic best threshold")
    print("-" * 40)

    print(
        f"Threshold: {best_threshold:.6f}"
    )

    print(
        f"F1:        {best_f1:.4f}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis threshold is diagnostic only."
        "\nIt must NOT be used as the final production threshold."
    )


# ============================================================
# Main
# ============================================================

for category in CATEGORIES:

    analyze_category(category)

print("\n")
print("=" * 70)
print("DECISION POLICY ANALYSIS COMPLETE")
print("=" * 70)