from src.production_inference import ProductionInferenceEngine
from src.validate_production_artifacts import main as validate_artifacts


def main():
    print("=" * 72)
    print("VISIONFORGE — PRODUCTION BUNDLE VALIDATION")
    print("=" * 72)
    print()

    print("STEP 1 — Artifact integrity")
    validate_artifacts()

    print()
    print("STEP 2 — Production engine startup")

    engine = ProductionInferenceEngine()

    expected_categories = {
        "bottle",
        "hazelnut",
        "cable",
        "capsule",
        "screw",
        "metal_nut",
    }

    loaded_categories = set(
        engine.reference_banks.keys()
    )

    if loaded_categories != expected_categories:
        raise RuntimeError(
            "Loaded categories do not match "
            "the production category set."
        )

    for category in expected_categories:
        if category not in engine.thresholds:
            raise RuntimeError(
                f"Missing threshold for {category}."
            )

    print(
        f"Loaded categories: "
        f"{sorted(loaded_categories)}"
    )

    print()
    print("PRODUCTION BUNDLE VALIDATION: PASS")


if __name__ == "__main__":
    main()