from src.production_inference import (
    ProductionInferenceEngine,
)

from src.production_config import (
    CATEGORIES,
)


def main():
    print("=" * 72)
    print("VISIONFORGE — PRODUCTION STARTUP TEST")
    print("=" * 72)

    print()
    print("Loading production inference engine...")
    print()

    engine = ProductionInferenceEngine(
        device="cpu",
    )

    print()
    print("Model: loaded")
    print(f"Device: {engine.device}")

    print()
    print("Reference banks:")

    for category in CATEGORIES:
        l4 = engine.reference_banks[
            category
        ]["L4"]

        l8 = engine.reference_banks[
            category
        ]["L8"]

        print(
            f"  {category:<12} "
            f"L4={tuple(l4.shape)} "
            f"L8={tuple(l8.shape)}"
        )

    print()
    print("Thresholds:")

    for category in CATEGORIES:
        threshold = engine.thresholds[
            category
        ]

        print(
            f"  {category:<12} "
            f"P99={threshold:.6f}"
        )

    print()
    print("=" * 72)
    print("PRODUCTION STARTUP: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()