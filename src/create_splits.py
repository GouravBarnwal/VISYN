from pathlib import Path
import json
import random


DATA_ROOT = Path("data")
OUTPUT_DIR = Path("artifacts/splits")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

SEED = 42
REFERENCE_RATIO = 0.80


def get_normal_images(category):
    train_good = DATA_ROOT / category / "train" / "good"

    if not train_good.exists():
        raise FileNotFoundError(f"Missing: {train_good}")

    return sorted(
        str(path.relative_to(DATA_ROOT))
        for path in train_good.glob("*.png")
    )


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_splits = {}

    for category in CATEGORIES:
        images = get_normal_images(category)

        shuffled = images.copy()
        random.shuffle(shuffled)

        split_index = int(len(shuffled) * REFERENCE_RATIO)

        reference = sorted(shuffled[:split_index])
        development = sorted(shuffled[split_index:])

        all_splits[category] = {
            "reference": reference,
            "development": development,
        }

        print("=" * 60)
        print(f"CATEGORY: {category}")
        print(f"Total normal: {len(images)}")
        print(f"Reference:    {len(reference)}")
        print(f"Development:  {len(development)}")

    output_file = OUTPUT_DIR / "normal_splits.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "reference_ratio": REFERENCE_RATIO,
                "categories": all_splits,
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print(f"Saved: {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()