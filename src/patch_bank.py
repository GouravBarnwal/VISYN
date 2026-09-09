from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


DATASET_ROOT = Path("data/bottle")


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

    # Remove average pooling and classification layers.
    model = torch.nn.Sequential(*list(model.children())[:-2])

    model.eval()

    return model


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature_map = model(image)

    # Shape:
    # (1, 512, 7, 7)
    feature_map = feature_map.squeeze(0)

    # Convert to:
    # (49, 512)
    patches = feature_map.permute(1, 2, 0).reshape(-1, 512)

    # Normalize each patch
    patches = patches / (
        torch.norm(patches, dim=1, keepdim=True) + 1e-8
    )

    return patches.numpy()


def main():
    model = load_model()

    train_dir = DATASET_ROOT / "train" / "good"

    patch_bank = []

    for image_path in sorted(train_dir.glob("*.png")):
        patches = extract_patches(model, image_path)
        patch_bank.append(patches)

    patch_bank = np.concatenate(patch_bank, axis=0)

    print("Training images:", len(list(train_dir.glob("*.png"))))
    print("Patches per image: 49")
    print("Patch dimension: 512")
    print("Total patches:", len(patch_bank))
    print("Patch bank shape:", patch_bank.shape)

    # Save for later use
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "bottle_patch_bank.npy"
    np.save(output_path, patch_bank)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()