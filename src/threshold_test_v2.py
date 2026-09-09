from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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
# Reference split
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

    reference_indices = indices[:split]

    return [
        images[i]
        for i in reference_indices
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
# Evaluate
# ============================================================

def evaluate_category(category):

    print("\n" + "=" * 70)
    print(f"FROZEN THRESHOLD TEST: {category}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load saved thresholds
    # --------------------------------------------------------

    threshold_file = (
        Path("artifacts")
        / category
        / "threshold_candidates_top5.npy"
    )

    p95, p99 = np.load(
        threshold_file
    )

    print(f"P95 threshold: {p95:.6f}")
    print(f"P99 threshold: {p99:.6f}")

    # --------------------------------------------------------
    # Build reference bank
    # --------------------------------------------------------

    reference_images = get_reference_images(
        category
    )

    print(
        f"Reference images: {len(reference_images)}"
    )

    all_patches = []

    for i, image_path in enumerate(
        reference_images
    ):

        all_patches.append(
            extract_features(image_path)
        )

        if (i + 1) % 50 == 0:
            print(
                f"Reference features: "
                f"{i + 1}/{len(reference_images)}"
            )

    patch_bank = np.concatenate(
        all_patches,
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
    # Load test set
    # --------------------------------------------------------

    good_images, defective_images = (
        get_test_images(category)
    )

    print(
        f"Test good:      {len(good_images)}"
    )

    print(
        f"Test defective: {len(defective_images)}"
    )

    # --------------------------------------------------------
    # Score all test images
    # --------------------------------------------------------

    scores = []
    labels = []

    for image_path in good_images:

        score = calculate_score(
            image_path,
            nn
        )

        scores.append(score)
        labels.append(0)

    for image_path in defective_images:

        score = calculate_score(
            image_path,
            nn
        )

        scores.append(score)
        labels.append(1)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    # --------------------------------------------------------
    # Three-way decision
    # --------------------------------------------------------

    predictions = []

    for score in scores:

        if score < p95:
            predictions.append("NORMAL")

        elif score < p99:
            predictions.append("REVIEW")

        else:
            predictions.append("DEFECT")

    predictions = np.asarray(
        predictions
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    normal_count = np.sum(
        predictions == "NORMAL"
    )

    review_count = np.sum(
        predictions == "REVIEW"
    )

    defect_count = np.sum(
        predictions == "DEFECT"
    )

    print("\nDecision distribution")
    print("-" * 45)

    print(
        f"NORMAL: {normal_count}"
    )

    print(
        f"REVIEW: {review_count}"
    )

    print(
        f"DEFECT: {defect_count}"
    )

    # --------------------------------------------------------
    # Good-image behavior
    # --------------------------------------------------------

    good_predictions = predictions[
        :len(good_images)
    ]

    good_normal = np.sum(
        good_predictions == "NORMAL"
    )

    good_review = np.sum(
        good_predictions == "REVIEW"
    )

    good_defect = np.sum(
        good_predictions == "DEFECT"
    )

    print("\nGOOD TEST IMAGES")
    print("-" * 45)

    print(
        f"NORMAL: {good_normal}"
    )

    print(
        f"REVIEW: {good_review}"
    )

    print(
        f"DEFECT: {good_defect}"
    )

    # --------------------------------------------------------
    # Defective-image behavior
    # --------------------------------------------------------

    defective_predictions = predictions[
        len(good_images):
    ]

    defective_normal = np.sum(
        defective_predictions == "NORMAL"
    )

    defective_review = np.sum(
        defective_predictions == "REVIEW"
    )

    defective_defect = np.sum(
        defective_predictions == "DEFECT"
    )

    print("\nDEFECTIVE TEST IMAGES")
    print("-" * 45)

    print(
        f"NORMAL: {defective_normal}"
    )

    print(
        f"REVIEW: {defective_review}"
    )

    print(
        f"DEFECT: {defective_defect}"
    )

    # --------------------------------------------------------
    # Binary evaluation
    #
    # NORMAL = acceptable
    # REVIEW + DEFECT = anomalous
    # --------------------------------------------------------

    binary_predictions = (
        predictions != "NORMAL"
    ).astype(int)

    binary_labels = labels

    cm = confusion_matrix(
        binary_labels,
        binary_predictions
    )

    precision = precision_score(
        binary_labels,
        binary_predictions,
        zero_division=0
    )

    recall = recall_score(
        binary_labels,
        binary_predictions,
        zero_division=0
    )

    f1 = f1_score(
        binary_labels,
        binary_predictions,
        zero_division=0
    )

    print("\nBinary evaluation")
    print("-" * 45)

    print(
        "Confusion matrix:"
    )

    print(cm)

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    good_scores = scores[
        :len(good_images)
    ]

    defective_scores = scores[
        len(good_images):
    ]

    print("\nScore distributions")
    print("-" * 45)

    print(
        f"Good median:      "
        f"{np.median(good_scores):.6f}"
    )

    print(
        f"Good maximum:     "
        f"{np.max(good_scores):.6f}"
    )

    print(
        f"Defect median:    "
        f"{np.median(defective_scores):.6f}"
    )

    print(
        f"Defect minimum:   "
        f"{np.min(defective_scores):.6f}"
    )

    return {
        "category": category,
        "p95": p95,
        "p99": p99,
        "good_normal": good_normal,
        "good_review": good_review,
        "good_defect": good_defect,
        "defect_normal": defective_normal,
        "defect_review": defective_review,
        "defect_defect": defective_defect,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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
print("FROZEN THRESHOLD SUMMARY")
print("=" * 70)

for r in results:

    print(f"\n{r['category']}")

    print(
        f"  Good → NORMAL:  "
        f"{r['good_normal']}"
    )

    print(
        f"  Good → REVIEW:  "
        f"{r['good_review']}"
    )

    print(
        f"  Good → DEFECT:  "
        f"{r['good_defect']}"
    )

    print(
        f"  Defect → NORMAL: "
        f"{r['defect_normal']}"
    )

    print(
        f"  Defect → REVIEW: "
        f"{r['defect_review']}"
    )

    print(
        f"  Defect → DEFECT: "
        f"{r['defect_defect']}"
    )

    print(
        f"  Precision: {r['precision']:.4f}"
    )

    print(
        f"  Recall:    {r['recall']:.4f}"
    )

    print(
        f"  F1:        {r['f1']:.4f}"
    )

print("\nTest complete.")