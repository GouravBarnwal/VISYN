from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


DATA_ROOT = Path("data")
SPLIT_FILE = Path("artifacts/splits/normal_splits.json")
OUTPUT_DIR = Path("artifacts/patch_banks/mobilenet_l8")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

DEVICE = torch.device("cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Layer-8 feature extractor: output = 48 x 14 x 14
    feature_extractor = torch.nn.Sequential(
        *list(model.features.children())[:9]
    )

    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    return feature_extractor


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        features = model(tensor)

    # (1, 48, 14, 14) -> (196, 48)
    features = features.squeeze(0)
    features = features.permute(1, 2, 0)
    features = features.reshape(-1, features.shape[-1])

    # L2 normalize descriptors
    features = torch.nn.functional.normalize(features, p=2, dim=1)

    return features.cpu().numpy().astype(np.float32)


def main():
    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()

    print("MobileNet L8 model loaded.")
    print(f"Device: {DEVICE}")
    print()

    for category in CATEGORIES:
        print("=" * 60)
        print(f"CATEGORY: {category}")

        reference_images = split_data["categories"][category]["reference"]

        all_patches = []

        for i, relative_path in enumerate(reference_images, start=1):
            image_path = DATA_ROOT / relative_path

            patches = extract_patches(model, image_path)
            all_patches.append(patches)

            if i % 50 == 0 or i == len(reference_images):
                print(
                    f"Reference features: "
                    f"{i}/{len(reference_images)}"
                )

        patch_bank = np.concatenate(all_patches, axis=0)

        output_file = OUTPUT_DIR / f"{category}.npy"
        np.save(output_file, patch_bank)

        print(f"Patch bank shape: {patch_bank.shape}")
        print(f"Saved: {output_file}")

    print("=" * 60)
    print("Clean MobileNet L8 patch banks created.")
    print("Done.")


if __name__ == "__main__":
    main()