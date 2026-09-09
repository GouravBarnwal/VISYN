from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms


DATASET_ROOT = Path("data/bottle")
PATCH_BANK_PATH = Path("artifacts/bottle_patch_bank.npy")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model = torch.nn.Sequential(
        *list(model.children())[:-2]
    )

    model.eval()

    return model


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature_map = model(tensor)

    feature_map = feature_map.squeeze(0)

    patches = feature_map.permute(
        1, 2, 0
    ).reshape(-1, 512)

    patches = patches / (
        torch.norm(
            patches,
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    return patches.numpy()


def get_anomaly_map(model, nn, image_path):
    patches = extract_patches(
        model,
        image_path
    )

    distances, _ = nn.kneighbors(patches)

    anomaly_map = distances.flatten().reshape(7, 7)

    return anomaly_map


def resize_anomaly_map(anomaly_map, size=(900, 900)):
    anomaly_map = anomaly_map.astype(np.float32)

    # Important:
    # Do NOT normalize each image independently here.
    # We want to preserve the original anomaly-score scale.
    anomaly_map = np.array(
        Image.fromarray(anomaly_map)
        .resize(
            size,
            Image.Resampling.BILINEAR
        )
    )

    return anomaly_map


def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")

    mask = np.array(mask)

    # MVTec masks use non-zero pixels for defect regions.
    mask = (mask > 0).astype(np.uint8)

    return mask


def main():

    print("Loading model...")

    model = load_model()

    print("Loading patch bank...")

    patch_bank = np.load(
        PATCH_BANK_PATH
    )

    print("Patch bank:", patch_bank.shape)

    # ---------------------------------------------------------
    # Build nearest-neighbor index
    # ---------------------------------------------------------

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(patch_bank)

    print("Nearest-neighbor index ready.")

    # ---------------------------------------------------------
    # Evaluate defect images
    # ---------------------------------------------------------

    defect_types = [
        "broken_large",
        "broken_small",
        "contamination",
    ]

    pixel_aurocs = []

    total_images = 0

    for defect_type in defect_types:

        image_dir = (
            DATASET_ROOT
            / "test"
            / defect_type
        )

        mask_dir = (
            DATASET_ROOT
            / "ground_truth"
            / defect_type
        )

        print(f"\nEvaluating: {defect_type}")

        for image_path in sorted(
            image_dir.glob("*.png")
        ):

            # MVTec naming:
            # 000.png → 000_mask.png
            mask_path = (
                mask_dir
                / f"{image_path.stem}_mask.png"
            )

            if not mask_path.exists():
                print(
                    "Missing mask:",
                    mask_path
                )
                continue

            anomaly_map = get_anomaly_map(
                model,
                nn,
                image_path
            )

            anomaly_map = resize_anomaly_map(
                anomaly_map
            )

            mask = load_mask(mask_path)

            # Flatten both into pixel-level samples.
            scores = anomaly_map.flatten()
            labels = mask.flatten()

            # Pixel AUROC requires both classes.
            if len(np.unique(labels)) < 2:
                continue

            pixel_auroc = roc_auc_score(
                labels,
                scores
            )

            pixel_aurocs.append(
                pixel_auroc
            )

            total_images += 1

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\nLocalization Evaluation")
    print("-----------------------")
    print(
        "Defect images evaluated:",
        total_images
    )

    if pixel_aurocs:
        print(
            f"Mean pixel AUROC: "
            f"{np.mean(pixel_aurocs):.4f}"
        )

        print(
            f"Minimum pixel AUROC: "
            f"{np.min(pixel_aurocs):.4f}"
        )

        print(
            f"Maximum pixel AUROC: "
            f"{np.max(pixel_aurocs):.4f}"
        )


if __name__ == "__main__":
    main()