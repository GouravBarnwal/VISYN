from pathlib import Path
import json

import numpy as np


# ============================================================
# CONFIG
# ============================================================

FUSION_PATH = Path(
    "artifacts/evaluation/day11_l4_l8_fusion.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/day11_l4_l8_bootstrap.json"
)

CATEGORIES = [
    "bottle",
    "hazelnut",
    "cable",
    "capsule",
    "screw",
    "metal_nut",
]

L4_WEIGHT = 0.5
L8_WEIGHT = 0.5

N_BOOTSTRAPS = 5000
SEED = 42


# ============================================================
# AUROC
# ============================================================

def calculate_auroc(
    normal_scores,
    defect_scores,
):
    normal_scores = np.asarray(
        normal_scores,
        dtype=np.float64,
    )

    defect_scores = np.asarray(
        defect_scores,
        dtype=np.float64,
    )

    if len(normal_scores) == 0:
        return np.nan

    if len(defect_scores) == 0:
        return np.nan

    comparisons = (
        defect_scores[:, None]
        - normal_scores[None, :]
    )

    greater = np.sum(
        comparisons > 0
    )

    ties = np.sum(
        comparisons == 0
    )

    return float(
        (
            greater
            + 0.5 * ties
        )
        / (
            len(defect_scores)
            * len(normal_scores)
        )
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def run_bootstrap(
    normal_l8,
    defect_l8,
    normal_fusion,
    defect_fusion,
    rng,
):
    normal_l8 = np.asarray(
        normal_l8,
        dtype=np.float64,
    )

    defect_l8 = np.asarray(
        defect_l8,
        dtype=np.float64,
    )

    normal_fusion = np.asarray(
        normal_fusion,
        dtype=np.float64,
    )

    defect_fusion = np.asarray(
        defect_fusion,
        dtype=np.float64,
    )

    n_normal = len(
        normal_l8
    )

    n_defect = len(
        defect_l8
    )

    # --------------------------------------------------------
    # Observed metrics
    # --------------------------------------------------------

    l8_auroc = calculate_auroc(
        normal_l8,
        defect_l8,
    )

    fusion_auroc = calculate_auroc(
        normal_fusion,
        defect_fusion,
    )

    observed_difference = (
        fusion_auroc
        - l8_auroc
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    differences = []

    for _ in range(
        N_BOOTSTRAPS
    ):
        normal_indices = rng.integers(
            0,
            n_normal,
            size=n_normal,
        )

        defect_indices = rng.integers(
            0,
            n_defect,
            size=n_defect,
        )

        sampled_normal_l8 = (
            normal_l8[
                normal_indices
            ]
        )

        sampled_defect_l8 = (
            defect_l8[
                defect_indices
            ]
        )

        sampled_normal_fusion = (
            normal_fusion[
                normal_indices
            ]
        )

        sampled_defect_fusion = (
            defect_fusion[
                defect_indices
            ]
        )

        bootstrap_l8 = calculate_auroc(
            sampled_normal_l8,
            sampled_defect_l8,
        )

        bootstrap_fusion = calculate_auroc(
            sampled_normal_fusion,
            sampled_defect_fusion,
        )

        if (
            np.isnan(bootstrap_l8)
            or np.isnan(bootstrap_fusion)
        ):
            continue

        differences.append(
            bootstrap_fusion
            - bootstrap_l8
        )

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    ci_low, ci_high = np.percentile(
        differences,
        [2.5, 97.5],
    )

    probability_positive = float(
        np.mean(
            differences > 0
        )
    )

    probability_nonnegative = float(
        np.mean(
            differences >= 0
        )
    )

    return {
        "l8_auroc": float(
            l8_auroc
        ),
        "fusion_auroc": float(
            fusion_auroc
        ),
        "observed_difference": float(
            observed_difference
        ),
        "bootstrap_samples": int(
            len(differences)
        ),
        "difference_ci_95": [
            float(ci_low),
            float(ci_high),
        ],
        "probability_fusion_beats_l8": (
            probability_positive
        ),
        "probability_fusion_at_least_l8": (
            probability_nonnegative
        ),
        "bootstrap_difference_mean": float(
            np.mean(differences)
        ),
        "bootstrap_difference_std": float(
            np.std(
                differences,
                ddof=1,
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "VISIONFORGE — DAY 11 L4 + L8 BOOTSTRAP"
    )
    print("=" * 72)

    print()
    print(
        "Comparison: 50/50 L4+L8 fusion vs L8"
    )
    print(
        f"L4 weight: {L4_WEIGHT:.1f}"
    )
    print(
        f"L8 weight: {L8_WEIGHT:.1f}"
    )
    print(
        f"Bootstrap samples: {N_BOOTSTRAPS}"
    )
    print(
        f"Seed: {SEED}"
    )
    print(
        "Split: canonical development"
    )
    print(
        "Final locked defect set: NOT USED"
    )
    print()

    # --------------------------------------------------------
    # Load fusion artifact
    # --------------------------------------------------------

    if not FUSION_PATH.exists():
        raise FileNotFoundError(
            f"Missing fusion artifact:\n"
            f"{FUSION_PATH}"
        )

    with open(
        FUSION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Verify experiment configuration
    # --------------------------------------------------------

    if data.get(
        "distance_metric"
    ) != "euclidean_cdist_p2":

        raise ValueError(
            "Expected Euclidean cdist scoring."
        )

    if data.get(
        "aggregation"
    ) != "TOP5":

        raise ValueError(
            "Expected TOP5 aggregation."
        )

    if data.get(
        "normalization"
    ) != "median_mad":

        raise ValueError(
            "Expected median/MAD normalization."
        )

    rng = np.random.default_rng(
        SEED
    )

    results = {
        "comparison": (
            "50/50 L4+L8 fusion vs L8"
        ),
        "l4_weight": L4_WEIGHT,
        "l8_weight": L8_WEIGHT,
        "bootstrap_samples": N_BOOTSTRAPS,
        "seed": SEED,
        "split": (
            "canonical_development"
        ),
        "final_set_used": False,
        "categories": {},
    }

    # ========================================================
    # CATEGORY LOOP
    # ========================================================

    all_l8_normal = []
    all_l8_defect = []
    all_fusion_normal = []
    all_fusion_defect = []

    for category in CATEGORIES:

        print("=" * 72)
        print(
            f"CATEGORY: {category}"
        )
        print("=" * 72)

        category_data = data[
            "categories"
        ][category]

        sample_scores = category_data[
            "sample_scores"
        ]

        # ----------------------------------------------------
        # Extract normalized scores
        # ----------------------------------------------------

        normal_l4 = np.asarray(
            sample_scores[
                "normal"
            ]["L4_normalized"],
            dtype=np.float64,
        )

        normal_l8 = np.asarray(
            sample_scores[
                "normal"
            ]["L8_normalized"],
            dtype=np.float64,
        )

        defect_l4 = np.asarray(
            sample_scores[
                "defect"
            ]["L4_normalized"],
            dtype=np.float64,
        )

        defect_l8 = np.asarray(
            sample_scores[
                "defect"
            ]["L8_normalized"],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Verify pairing
        # ----------------------------------------------------

        if len(normal_l4) != len(
            normal_l8
        ):
            raise ValueError(
                f"{category}: normal L4/L8 "
                "length mismatch."
            )

        if len(defect_l4) != len(
            defect_l8
        ):
            raise ValueError(
                f"{category}: defect L4/L8 "
                "length mismatch."
            )

        # ----------------------------------------------------
        # Construct 50/50 fusion
        # ----------------------------------------------------

        normal_fusion = (
            L4_WEIGHT * normal_l4
            + L8_WEIGHT * normal_l8
        )

        defect_fusion = (
            L4_WEIGHT * defect_l4
            + L8_WEIGHT * defect_l8
        )

        # ----------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------

        bootstrap = run_bootstrap(
            normal_l8,
            defect_l8,
            normal_fusion,
            defect_fusion,
            rng,
        )

        results["categories"][
            category
        ] = {
            "normal_count": int(
                len(normal_l8)
            ),
            "defect_count": int(
                len(defect_l8)
            ),
            **bootstrap,
        }

        print(
            f"Normal samples: "
            f"{len(normal_l8)}"
        )

        print(
            f"Defect samples: "
            f"{len(defect_l8)}"
        )

        print(
            f"L8 AUROC: "
            f"{bootstrap['l8_auroc']:.4f}"
        )

        print(
            f"50/50 Fusion AUROC: "
            f"{bootstrap['fusion_auroc']:.4f}"
        )

        print(
            f"Observed difference: "
            f"{bootstrap['observed_difference']:+.4f}"
        )

        print(
            "95% CI: "
            f"["
            f"{bootstrap['difference_ci_95'][0]:+.4f}, "
            f"{bootstrap['difference_ci_95'][1]:+.4f}"
            f"]"
        )

        print(
            "P(fusion > L8): "
            f"{bootstrap['probability_fusion_beats_l8']:.4f}"
        )

        print()

        # ----------------------------------------------------
        # Pooled sample collection
        #
        # These are NOT used as the primary model-selection
        # metric. They are included only as an additional
        # diagnostic.
        # ----------------------------------------------------

        all_l8_normal.extend(
            normal_l8.tolist()
        )

        all_l8_defect.extend(
            defect_l8.tolist()
        )

        all_fusion_normal.extend(
            normal_fusion.tolist()
        )

        all_fusion_defect.extend(
            defect_fusion.tolist()
        )

    # ========================================================
    # POOLED BOOTSTRAP DIAGNOSTIC
    # ========================================================

    print("=" * 72)
    print(
        "POOLED DIAGNOSTIC"
    )
    print("=" * 72)

    pooled = run_bootstrap(
        np.asarray(
            all_l8_normal,
            dtype=np.float64,
        ),
        np.asarray(
            all_l8_defect,
            dtype=np.float64,
        ),
        np.asarray(
            all_fusion_normal,
            dtype=np.float64,
        ),
        np.asarray(
            all_fusion_defect,
            dtype=np.float64,
        ),
        rng,
    )

    results["pooled_diagnostic"] = pooled

    print(
        f"L8 AUROC: "
        f"{pooled['l8_auroc']:.4f}"
    )

    print(
        f"50/50 Fusion AUROC: "
        f"{pooled['fusion_auroc']:.4f}"
    )

    print(
        f"Observed difference: "
        f"{pooled['observed_difference']:+.4f}"
    )

    print(
        "95% CI: "
        f"["
        f"{pooled['difference_ci_95'][0]:+.4f}, "
        f"{pooled['difference_ci_95'][1]:+.4f}"
        f"]"
    )

    print()

    # ========================================================
    # SUMMARY
    # ========================================================

    category_differences = [
        results[
            "categories"
        ][category][
            "observed_difference"
        ]
        for category in CATEGORIES
    ]

    category_ci = [
        results[
            "categories"
        ][category][
            "difference_ci_95"
        ]
        for category in CATEGORIES
    ]

    results["summary"] = {
        "macro_observed_difference": float(
            np.mean(
                category_differences
            )
        ),
        "categories_with_positive_difference": int(
            np.sum(
                np.asarray(
                    category_differences
                )
                > 0
            )
        ),
        "categories_with_ci_entirely_above_zero": int(
            np.sum(
                [
                    ci[0] > 0
                    for ci in category_ci
                ]
            )
        ),
    }

    # ========================================================
    # SAVE
    # ========================================================

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

    print("=" * 72)
    print(
        "BOOTSTRAP SUMMARY"
    )
    print("=" * 72)

    print(
        f"Macro mean AUROC difference: "
        f"{results['summary']['macro_observed_difference']:+.4f}"
    )

    print(
        "Categories where fusion improved: "
        f"{results['summary']['categories_with_positive_difference']}"
        f"/{len(CATEGORIES)}"
    )

    print(
        "Categories with CI entirely above zero: "
        f"{results['summary']['categories_with_ci_entirely_above_zero']}"
        f"/{len(CATEGORIES)}"
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()