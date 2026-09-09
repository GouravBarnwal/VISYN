from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms


DATA_ROOT = Path("data")
SPLIT_FILE = Path("artifacts/splits/normal_splits.json")
BANK_DIR = Path("artifacts/patch_banks/mobilenet_l8")

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

    extractor = torch.nn.Sequential(
        *list(model.features.children())[:9]
    )

    extractor.eval()
    extractor.to(DEVICE)

    return extractor


def extract_patches(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        features = model(tensor)

    # (1, 48, 14, 14) -> (196, 48)
    features = features.squeeze(0)
    features = features.permute(1, 2, 0)
    features = features.reshape(-1, features.shape[-1])

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )

    return features.cpu().numpy().astype(np.float32)


def score_image(model, image_path, nn_index):
    patches = extract_patches(model, image_path)

    distances, _ = nn_index.kneighbors(patches)

    distances = distances[:, 0]

    max_score = float(np.max(distances))

    top5 = np.sort(distances)[-5:]
    top5_score = float(np.mean(top5))

    top10 = np.sort(distances)[-10:]
    top10_score = float(np.mean(top10))

    return max_score, top5_score, top10_score


def get_defect_images(category):
    test_root = DATA_ROOT / category / "test"

    images = []

    for defect_type in sorted(test_root.iterdir()):
        if not defect_type.is_dir():
            continue

        for image_path in sorted(defect_type.glob("*.png")):
            images.append(image_path)

    return images


def main():
    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    model = load_model()

    print("Clean development evaluation")
    print("Reference = 80% normal train")
    print("Development = 20% normal train")
    print("Defect data is NOT used for selecting aggregation.")
    print()

    all_results = []

    for category in CATEGORIES:
        print("=" * 60)
        print(f"CATEGORY: {category}")

        bank_path = BANK_DIR / f"{category}.npy"
        patch_bank = np.load(bank_path)

        print(f"Patch bank: {patch_bank.shape}")

        nn_index = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
            algorithm="auto",
        )

        nn_index.fit(patch_bank)

        development_images = split_data["categories"][category]["development"]

        max_scores = []
        top5_scores = []
        top10_scores = []

        for i, relative_path in enumerate(
            development_images,
            start=1,
        ):
            image_path = DATA_ROOT / relative_path

            max_score, top5_score, top10_score = score_image(
                model,
                image_path,
                nn_index,
            )

            max_scores.append(max_score)
            top5_scores.append(top5_score)
            top10_scores.append(top10_score)

            if i % 20 == 0 or i == len(development_images):
                print(
                    f"Development images: "
                    f"{i}/{len(development_images)}"
                )

        # --------------------------------------------------
        # Exploratory defect comparison
        # --------------------------------------------------
        #
        # We do NOT use this to select the final aggregation.
        # It is reported only so we can see whether the
        # development-normal score distributions separate
        # from known defects.
        #
        defect_images = get_defect_images(category)

        defect_max = []
        defect_top5 = []
        defect_top10 = []

        for image_path in defect_images:
            scores = score_image(
                model,
                image_path,
                nn_index,
            )

            defect_max.append(scores[0])
            defect_top5.append(scores[1])
            defect_top10.append(scores[2])

        y_true = np.array(
            [0] * len(max_scores) +
            [1] * len(defect_max)
        )

        results = {}

        for name, normal, defect in [
            ("MAX", max_scores, defect_max),
            ("TOP5", top5_scores, defect_top5),
            ("TOP10", top10_scores, defect_top10),
        ]:
            scores = np.array(normal + defect)

            auc = roc_auc_score(y_true, scores)

            results[name] = auc

        print()
        print("## Development Results")
        print(
            f"MAX:         {results['MAX']:.4f}"
        )
        print(
            f"TOP-5 MEAN:  {results['TOP5']:.4f}"
        )
        print(
            f"TOP-10 MEAN: {results['TOP10']:.4f}"
        )

        all_results.append(
            {
                "category": category,
                **results,
            }
        )

    print()
    print("=" * 60)
    print("CLEAN DEVELOPMENT AGGREGATION COMPARISON")
    print("=" * 60)
    print()
    print(
        f"{'Category':<12}"
        f"{'MAX':>10}"
        f"{'TOP5':>10}"
        f"{'TOP10':>10}"
    )

    for result in all_results:
        print(
            f"{result['category']:<12}"
            f"{result['MAX']:>10.4f}"
            f"{result['TOP5']:>10.4f}"
            f"{result['TOP10']:>10.4f}"
        )

    mean_max = np.mean([r["MAX"] for r in all_results])
    mean_top5 = np.mean([r["TOP5"] for r in all_results])
    mean_top10 = np.mean([r["TOP10"] for r in all_results])

    print("-" * 42)
    print(
        f"{'MEAN':<12}"
        f"{mean_max:>10.4f}"
        f"{mean_top5:>10.4f}"
        f"{mean_top10:>10.4f}"
    )

    output_path = Path(
        "artifacts/clean_development_aggregation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            {
                "reference_ratio": 0.80,
                "development_ratio": 0.20,
                "seed": 42,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print()
    print(f"Saved: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()