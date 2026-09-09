from pathlib import Path
import json
import random

DATA_ROOT = Path("data")
OUTPUT_FILE = Path("artifacts/splits/defect_splits.json")

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

SEED = 42
DEVELOPMENT_RATIO = 0.50


def main():
    rng = random.Random(SEED)

    all_splits = {}

    for category in CATEGORIES:
        test_root = DATA_ROOT / category / "test"

        development = []
        final_test = []

        defect_types = sorted(
            p.name
            for p in test_root.iterdir()
            if p.is_dir() and p.name != "good"
        )

        print("=" * 60)
        print(f"CATEGORY: {category}")

        for defect_type in defect_types:
            images = sorted(
                str(p.relative_to(DATA_ROOT))
                for p in (test_root / defect_type).glob("*.png")
            )

            shuffled = images.copy()

            # Deterministic but independent per defect type
            rng.shuffle(shuffled)

            split_index = int(
                len(shuffled) * DEVELOPMENT_RATIO
            )

            dev_images = sorted(shuffled[:split_index])
            final_images = sorted(shuffled[split_index:])

            development.extend(dev_images)
            final_test.extend(final_images)

            print(
                f"{defect_type:<25} "
                f"total={len(images):3d} "
                f"dev={len(dev_images):3d} "
                f"final={len(final_images):3d}"
            )

        all_splits[category] = {
            "development": sorted(development),
            "final_test": sorted(final_test),
        }

        print(
            f"TOTAL{'':20} "
            f"dev={len(development):3d} "
            f"final={len(final_test):3d}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "development_ratio": DEVELOPMENT_RATIO,
                "categories": all_splits,
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print(f"Saved: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main();