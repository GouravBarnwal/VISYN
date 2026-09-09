from pathlib import Path
import json

import torch

from src.production_inference import (
    ProductionInferenceEngine,
)

from src.production_config import (
    CATEGORIES,
    DATA_ROOT,
    SCORE_EQUIVALENCE_TOLERANCE,
    SPLITS_PATH,
)


OUTPUT_PATH = (
    Path("artifacts/evaluation")
    / "day17_production_engine_validation.json"
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@torch.no_grad()
def score_reference(
    engine,
    image_path,
    category,
):
    l4_queries, l8_queries = (
        engine.extract_features(
            [image_path]
        )
    )

    l4_bank = engine.reference_banks[
        category
    ]["L4"]

    l8_bank = engine.reference_banks[
        category
    ]["L8"]

    l4_score = engine._score_chunked(
        l4_queries[0],
        l4_bank,
    )

    l8_score = engine._score_chunked(
        l8_queries[0],
        l8_bank,
    )

    fusion = (
        0.5 * l4_score
        + 0.5 * l8_score
    )

    threshold = engine.thresholds[
        category
    ]

    decision, review_threshold = (
        engine._classify(
            fusion.item(),
            threshold,
        )
    )

    return {
        "l4": float(
            l4_score.item()
        ),
        "l8": float(
            l8_score.item()
        ),
        "fusion": float(
            fusion.item()
        ),
        "threshold": float(
            threshold
        ),
        "review_threshold": float(
            review_threshold
        ),
        "decision": decision,
    }


def main():
    print("=" * 72)
    print(
        "VISIONFORGE — DAY 17"
    )
    print(
        "PRODUCTION ENGINE VALIDATION"
    )
    print("=" * 72)

    split = load_json(
        SPLITS_PATH
    )

    engine = ProductionInferenceEngine(
        device="cpu",
    )

    results = {
        "architecture": (
            "MobileNetV3-Small L4+L8"
        ),
        "fusion": "raw_50_50",
        "aggregation": "TOP5",
        "chunk_size": 8192,
        "categories": {},
    }

    global_max_l4 = 0.0
    global_max_l8 = 0.0
    global_max_fusion = 0.0

    decision_matches = 0
    decision_total = 0

    for category in CATEGORIES:
        print()
        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        development_paths = split[
            "categories"
        ][category]["development"]

        test_paths = development_paths[:5]

        category_results = {
            "samples": []
        }

        for relative_path in test_paths:
            image_path = (
                DATA_ROOT
                / relative_path
            )

            # ------------------------------------------------
            # New production engine
            # ------------------------------------------------

            production_result = (
                engine.inspect(
                    image_path,
                    category,
                )
            )

            # ------------------------------------------------
            # Independent reference calculation
            # using the exact same loaded artifacts
            # ------------------------------------------------

            reference_result = (
                score_reference(
                    engine,
                    image_path,
                    category,
                )
            )

            l4_difference = abs(
                production_result[
                    "l4_score"
                ]
                - reference_result["l4"]
            )

            l8_difference = abs(
                production_result[
                    "l8_score"
                ]
                - reference_result["l8"]
            )

            fusion_difference = abs(
                production_result[
                    "anomaly_score"
                ]
                - reference_result["fusion"]
            )

            decision_match = (
                production_result[
                    "decision"
                ]
                == reference_result[
                    "decision"
                ]
            )

            global_max_l4 = max(
                global_max_l4,
                l4_difference,
            )

            global_max_l8 = max(
                global_max_l8,
                l8_difference,
            )

            global_max_fusion = max(
                global_max_fusion,
                fusion_difference,
            )

            decision_total += 1

            if decision_match:
                decision_matches += 1

            print()
            print(
                relative_path
            )
            print(
                f"  L4 difference: "
                f"{l4_difference:.10f}"
            )
            print(
                f"  L8 difference: "
                f"{l8_difference:.10f}"
            )
            print(
                f"  Fusion difference: "
                f"{fusion_difference:.10f}"
            )
            print(
                f"  Decision: "
                f"{production_result['decision']}"
            )
            print(
                f"  Decision match: "
                f"{decision_match}"
            )

            category_results[
                "samples"
            ].append(
                {
                    "path": str(
                        relative_path
                    ),
                    "production": (
                        production_result
                    ),
                    "reference": (
                        reference_result
                    ),
                    "l4_difference": (
                        l4_difference
                    ),
                    "l8_difference": (
                        l8_difference
                    ),
                    "fusion_difference": (
                        fusion_difference
                    ),
                    "decision_match": (
                        decision_match
                    ),
                }
            )

        results[
            "categories"
        ][category] = category_results

    equivalence_pass = (
        global_max_l4
        < SCORE_EQUIVALENCE_TOLERANCE
        and global_max_l8
        < SCORE_EQUIVALENCE_TOLERANCE
        and global_max_fusion
        < SCORE_EQUIVALENCE_TOLERANCE
        and decision_matches
        == decision_total
    )

    results[
        "global_max_l4_difference"
    ] = global_max_l4

    results[
        "global_max_l8_difference"
    ] = global_max_l8

    results[
        "global_max_fusion_difference"
    ] = global_max_fusion

    results[
        "decision_matches"
    ] = decision_matches

    results[
        "decision_total"
    ] = decision_total

    results[
        "equivalence_pass"
    ] = equivalence_pass

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
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("FINAL VALIDATION")
    print("=" * 72)

    print(
        "L4 numerical equivalence: "
        f"{global_max_l4:.10f}"
    )

    print(
        "L8 numerical equivalence: "
        f"{global_max_l8:.10f}"
    )

    print(
        "Fusion numerical equivalence: "
        f"{global_max_fusion:.10f}"
    )

    print(
        f"Decision matches: "
        f"{decision_matches}/{decision_total}"
    )

    print()

    if equivalence_pass:
        print(
            "PRODUCTION ENGINE: PASS"
        )
    else:
        print(
            "PRODUCTION ENGINE: FAIL"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()