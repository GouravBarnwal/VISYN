import json
from pathlib import Path


COMPARISON_PATH = Path(
    "artifacts/evaluation/mobilenet_vs_dino_dev_comparison.json"
)

FUSION_PATH = Path(
    "artifacts/evaluation/fusion_dev_results.json"
)

BOOTSTRAP_PATH = Path(
    "artifacts/evaluation/fusion_bootstrap_results.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/final_model_selection_dev.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 72)
    print("VISIONFORGE — FINAL DEVELOPMENT MODEL SELECTION")
    print("=" * 72)

    comparison = load_json(COMPARISON_PATH)
    fusion = load_json(FUSION_PATH)
    bootstrap = load_json(BOOTSTRAP_PATH)

    # -------------------------------------------------------------
    # Categories used by the locked development protocol
    # -------------------------------------------------------------

    categories = [
        "bottle",
        "hazelnut",
        "cable",
        "capsule",
        "screw",
        "metal_nut",
    ]

    methods = [
        "max",
        "top5",
        "top10",
    ]

    # -------------------------------------------------------------
    # Standalone model results
    #
    # The comparison JSON stores:
    #
    # results
    #   ├── mobilenet
    #   │     └── category
    #   │           ├── max
    #   │           ├── top5
    #   │           └── top10
    #   └── dino
    #         └── category
    #               ├── max
    #               ├── top5
    #               └── top10
    #
    # -------------------------------------------------------------

    standalone = {
        "MobileNet L8": {},
        "DINOv2": {},
    }

    for method in methods:

        standalone["MobileNet L8"][method] = float(
            sum(
                comparison["results"]["mobilenet"][category][method]
                for category in categories
            )
            / len(categories)
        )

        standalone["DINOv2"][method] = float(
            sum(
                comparison["results"]["dino"][category][method]
                for category in categories
            )
            / len(categories)
        )

    # -------------------------------------------------------------
    # Best global fusion result for each aggregation
    # -------------------------------------------------------------

    fusion_best = {}

    for method in methods:

        best = fusion[method]["best_global"]

        fusion_best[method] = {
            "mobile_weight": float(
                best["mobile_weight"]
            ),
            "dino_weight": float(
                best["dino_weight"]
            ),
            "macro_auroc": float(
                best["macro_auroc"]
            ),
        }

    # -------------------------------------------------------------
    # Consolidated development table
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("DEVELOPMENT MODEL COMPARISON")
    print("=" * 72)

    print()
    print(
        f"{'Model':<22}"
        f"{'Aggregation':<14}"
        f"{'Macro AUROC':>14}"
    )

    print("-" * 52)

    for model in [
        "MobileNet L8",
        "DINOv2",
    ]:

        for method in methods:

            print(
                f"{model:<22}"
                f"{method.upper():<14}"
                f"{standalone[model][method]:>14.4f}"
            )

    for method in methods:

        result = fusion_best[method]

        print(
            f"{'Fusion':<22}"
            f"{method.upper():<14}"
            f"{result['macro_auroc']:>14.4f}"
        )

    # -------------------------------------------------------------
    # Find strongest standalone configuration
    # -------------------------------------------------------------

    standalone_candidates = []

    for model, model_results in standalone.items():

        for method, auc in model_results.items():

            standalone_candidates.append(
                {
                    "model": model,
                    "aggregation": method,
                    "macro_auroc": float(auc),
                }
            )

    best_standalone = max(
        standalone_candidates,
        key=lambda x: x["macro_auroc"],
    )

    # -------------------------------------------------------------
    # Find strongest global fusion configuration
    # -------------------------------------------------------------

    fusion_candidates = []

    for method, result in fusion_best.items():

        fusion_candidates.append(
            {
                "aggregation": method,
                "mobile_weight": result["mobile_weight"],
                "dino_weight": result["dino_weight"],
                "macro_auroc": result["macro_auroc"],
            }
        )

    best_fusion = max(
        fusion_candidates,
        key=lambda x: x["macro_auroc"],
    )

    # -------------------------------------------------------------
    # Bootstrap evidence
    # -------------------------------------------------------------

    bootstrap_pooled = bootstrap["pooled"]

    # -------------------------------------------------------------
    # Production candidate
    #
    # We deliberately select the strongest standalone model rather
    # than the tiny fusion improvement because the bootstrap analysis
    # did not establish a robust advantage for fusion.
    # -------------------------------------------------------------

    production_candidate = {
        "model": best_standalone["model"],
        "aggregation": best_standalone["aggregation"],
        "macro_development_auroc": (
            best_standalone["macro_auroc"]
        ),
        "decision": "selected_for_next_stage",
        "reason": (
            "Strongest standalone development configuration. "
            "Fusion provided only a very small development "
            "improvement and did not demonstrate robust stability."
        ),
    }

    # -------------------------------------------------------------
    # Build final consolidated artifact
    # -------------------------------------------------------------

    output = {
        "protocol": comparison["protocol"],
        "standalone": standalone,
        "fusion_best": fusion_best,
        "best_standalone": best_standalone,
        "best_fusion": best_fusion,
        "fusion_bootstrap_pooled": bootstrap_pooled,
        "production_candidate": production_candidate,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("BEST STANDALONE")
    print("=" * 72)

    print(
        f"Model:       "
        f"{best_standalone['model']}"
    )

    print(
        f"Aggregation: "
        f"{best_standalone['aggregation'].upper()}"
    )

    print(
        f"Macro AUROC: "
        f"{best_standalone['macro_auroc']:.4f}"
    )

    print()
    print("=" * 72)
    print("BEST GLOBAL FUSION")
    print("=" * 72)

    print(
        f"Aggregation: "
        f"{best_fusion['aggregation'].upper()}"
    )

    print(
        f"MobileNet weight: "
        f"{best_fusion['mobile_weight']:.1f}"
    )

    print(
        f"DINOv2 weight:    "
        f"{best_fusion['dino_weight']:.1f}"
    )

    print(
        f"Macro AUROC:      "
        f"{best_fusion['macro_auroc']:.4f}"
    )

    print()
    print("=" * 72)
    print("FUSION BOOTSTRAP")
    print("=" * 72)

    print(
        f"MobileNet: "
        f"{bootstrap_pooled['mobile_auroc']:.4f}"
    )

    print(
        f"Fusion:    "
        f"{bootstrap_pooled['fusion_auroc']:.4f}"
    )

    print(
        f"Gain:      "
        f"{bootstrap_pooled['difference']:+.4f}"
    )

    print()
    print("=" * 72)
    print("SELECTED FOR NEXT STAGE")
    print("=" * 72)

    print(
        f"{production_candidate['model']} + "
        f"{production_candidate['aggregation'].upper()}"
    )

    print(
        f"Development Macro AUROC: "
        f"{production_candidate['macro_development_auroc']:.4f}"
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()