from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


DATASET_ROOT = Path("data/bottle")


def extract_feature(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (256, 256))

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [32],
        [0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        histogram,
    ).flatten()

    return histogram


def main():
    # --------------------------------------------------
    # 1. Build normal reference from training images
    # --------------------------------------------------

    train_dir = DATASET_ROOT / "train" / "good"

    train_features = []

    for image_path in train_dir.glob("*.png"):
        feature = extract_feature(image_path)
        train_features.append(feature)

    train_features = np.array(train_features)

    reference = train_features.mean(axis=0)

    print("Training images:", len(train_features))
    print("Feature dimension:", reference.shape[0])

    # --------------------------------------------------
    # 2. Evaluate on test data
    # --------------------------------------------------

    test_groups = {
        "good": 0,
        "broken_large": 1,
        "broken_small": 1,
        "contamination": 1,
    }

    scores = []
    labels = []

    for group_name, label in test_groups.items():

        folder = DATASET_ROOT / "test" / group_name

        for image_path in folder.glob("*.png"):

            feature = extract_feature(image_path)

            anomaly_score = np.linalg.norm(
                feature - reference
            )

            scores.append(anomaly_score)
            labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    # --------------------------------------------------
    # 3. AUROC
    # --------------------------------------------------

    auroc = roc_auc_score(labels, scores)

    print("\nEvaluation")
    print("----------")
    print("Test images:", len(scores))
    print(f"AUROC: {auroc:.4f}")

    # --------------------------------------------------
    # 4. Choose a simple threshold
    # --------------------------------------------------

    threshold = np.median(
        scores[labels == 0]
    )

    predictions = (scores >= threshold).astype(int)

    print(f"\nThreshold: {threshold:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(labels, predictions))

    print("\nClassification Report")
    print(
        classification_report(
            labels,
            predictions,
            target_names=["normal", "defect"],
        )
    )


if __name__ == "__main__":
    main()