from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Configuration
# ============================================================

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

    # Expected:
    # (1, 48, 14, 14)

    features = features.squeeze(0)

    # (48, 14, 14)
    # ->
    # (196, 48)

    patches = features.permute(
        1, 2, 0
    ).reshape(
        -1,
        features.shape[0]
    )

    return patches.numpy()


# ============================================================
# Build reference / calibration split
# ============================================================

def split_normal_images(category):

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

    reference_indices = indices[:split]
    calibration_indices = indices[split:]

    reference_images = [
        images[i]
        for i in reference_indices
    ]

    calibration_images = [
        images[i]
        for i in calibration_indices
    ]

    return reference_images, calibration_images


# ============================================================
# Build patch bank
# ============================================================

def build_patch_bank(images):

    patches = []

    for i, image_path in enumerate(images):

        features = extract_features(image_path)

        patches.append(features)

        if (i + 1) % 50 == 0:
            print(
                f"  features: "
                f"{i + 1}/{len(images)}"
            )

    return np.concatenate(
        patches,
        axis=0
    )


# ============================================================
# TOP-5 anomaly score
# ============================================================

def top5_score(distances):

    distances = np.asarray(distances)

    return np.mean(
        np.sort(distances)[-5:]
    )


# ============================================================
# Calculate scores
# ============================================================

def calculate_scores(images, nn):

    scores = []

    for image_path in images:

        patches = extract_features(
            image_path
        )

        distances, _ = nn.kneighbors(
            patches,
            return_distance=True
        )

        distances = distances.ravel()

        score = top5_score(distances)

        scores.append(score)

    return np.asarray(scores)


# ============================================================
# Calibration
# ============================================================

def calibrate_category(category):

    print("\n" + "=" * 65)
    print(f"CALIBRATION: {category}")
    print("=" * 65)

    reference_images, calibration_images = (
        split_normal_images(category)
    )

    print(
        f"Total normal train images: "
        f"{len(reference_images) + len(calibration_images)}"
    )

    print(
        f"Reference images:           "
        f"{len(reference_images)}"
    )

    print(
        f"Calibration images:          "
        f"{len(calibration_images)}"
    )

    # --------------------------------------------------------
    # Build reference bank
    # --------------------------------------------------------

    print("\nBuilding patch bank...")

    patch_bank = build_patch_bank(
        reference_images
    )

    print(
        "Patch bank shape:",
        patch_bank.shape
    )

    # --------------------------------------------------------
    # Nearest-neighbor index
    # --------------------------------------------------------

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    # --------------------------------------------------------
    # Calibration scores
    # --------------------------------------------------------

    print("\nCalculating calibration scores...")

    scores = calculate_scores(
        calibration_images,
        nn
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    p90 = np.percentile(scores, 90)
    p95 = np.percentile(scores, 95)
    p97 = np.percentile(scores, 97)
    p98 = np.percentile(scores, 98)
    p99 = np.percentile(scores, 99)

    print("\nCalibration score distribution")
    print("-" * 45)

    print(f"Minimum: {scores.min():.6f}")
    print(f"Mean:    {scores.mean():.6f}")
    print(f"Median:  {np.median(scores):.6f}")
    print(f"P90:     {p90:.6f}")
    print(f"P95:     {p95:.6f}")
    print(f"P97:     {p97:.6f}")
    print(f"P98:     {p98:.6f}")
    print(f"P99:     {p99:.6f}")
    print(f"Maximum: {scores.max():.6f}")

    # --------------------------------------------------------
    # Candidate operating policy
    # --------------------------------------------------------

    print("\nCandidate threshold policy")
    print("-" * 45)

    print(
        f"NORMAL:  score < P95  ({p95:.6f})"
    )

    print(
        f"REVIEW:  P95 <= score < P99 "
        f"({p95:.6f} - {p99:.6f})"
    )

    print(
        f"DEFECT:  score >= P99 "
        f"({p99:.6f})"
    )

    # --------------------------------------------------------
    # Save calibration results
    # --------------------------------------------------------

    output_dir = Path("artifacts") / category

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "calibration_scores_top5.npy",
        scores
    )

    np.save(
        output_dir / "threshold_candidates_top5.npy",
        np.array([
            p95,
            p99
        ])
    )

    print(
        "\nSaved calibration artifacts to:",
        output_dir
    )

    return {
        "category": category,
        "p95": p95,
        "p99": p99,
    }


# ============================================================
# Main
# ============================================================

results = []

for category in CATEGORIES:

    result = calibrate_category(
        category
    )

    results.append(result)


# ============================================================
# Summary
# ============================================================

print("\n\n" + "=" * 65)
print("CALIBRATION SUMMARY")
print("=" * 65)

for result in results:

    print(
        f"{result['category']:<12}"
        f"P95={result['p95']:.6f}    "
        f"P99={result['p99']:.6f}"
    )

print("\nCalibration complete.")