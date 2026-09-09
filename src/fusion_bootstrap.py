import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


DINO_PATH = Path(
    "artifacts/evaluation/dino_dev_scores_normalized.json"
)

MOBILE_PATH = Path(
    "artifacts/evaluation/mobilenet_l8_dev_scores.json"
)

FUSION_PATH = Path(
    "artifacts/evaluation/fusion_dev_results.json"
)

OUTPUT_PATH = Path(
    "artifacts/evaluation/fusion_bootstrap_results.json"
)

N_BOOTSTRAPS = 5000
SEED = 42

MOBILE_WEIGHT = 0.1
DINO_WEIGHT = 0.9


def normalize_path(path):
    return Path(
        path.replace("\\", "/")
    ).as_posix()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_score(item):
    return float(item["max"])


def build_lookup(items):
    return {
        normalize_path(item["path"]): item
        for item in items
    }


def robust_scale(scores):
    median = np.median(scores)

    mad = np.median(
        np.abs(scores - median)
    )

    scale = 1.4826 * mad

    if scale < 1e-12:
        scale = np.std(scores)

    if scale < 1e-12:
        scale = 1.0

    return median, scale


def normalize_scores(scores, median, scale):
    return (scores - median) / scale


def prepare_category(dino, mobile, category):

    dino_normal = build_lookup(
        dino[category]["normal_dev"]
    )

    dino_defect = build_lookup(
        dino[category]["defect_dev"]
    )

    mobile_normal = build_lookup(
        mobile[category]["normal_dev"]
    )

    mobile_defect = build_lookup(
        mobile[category]["defect_dev"]
    )

    if set(dino_normal) != set(mobile_normal):
        raise RuntimeError(
            f"Normal image mismatch: {category}"
        )

    if set(dino_defect) != set(mobile_defect):
        raise RuntimeError(
            f"Defect image mismatch: {category}"
        )

    normal_paths = sorted(dino_normal)
    defect_paths = sorted(dino_defect)

    dino_normal_raw = np.array(
        [
            get_score(dino_normal[p])
            for p in normal_paths
        ],
        dtype=np.float64,
    )

    dino_defect_raw = np.array(
        [
            get_score(dino_defect[p])
            for p in defect_paths
        ],
        dtype=np.float64,
    )

    mobile_normal_raw = np.array(
        [
            get_score(mobile_normal[p])
            for p in normal_paths
        ],
        dtype=np.float64,
    )

    mobile_defect_raw = np.array(
        [
            get_score(mobile_defect[p])
            for p in defect_paths
        ],
        dtype=np.float64,
    )

    # Normalization is fitted ONLY using normal_dev.
    dino_median, dino_scale = robust_scale(
        dino_normal_raw
    )

    mobile_median, mobile_scale = robust_scale(
        mobile_normal_raw
    )

    dino_normal = normalize_scores(
        dino_normal_raw,
        dino_median,
        dino_scale,
    )

    dino_defect = normalize_scores(
        dino_defect_raw,
        dino_median,
        dino_scale,
    )

    mobile_normal = normalize_scores(
        mobile_normal_raw,
        mobile_median,
        mobile_scale,
    )

    mobile_defect = normalize_scores(
        mobile_defect_raw,
        mobile_median,
        mobile_scale,
    )

    return (
        mobile_normal,
        mobile_defect,
        dino_normal,
        dino_defect,
    )


def bootstrap_category(
    mobile_normal,
    mobile_defect,
    dino_normal,
    dino_defect,
    rng,
):

    normal_n = len(mobile_normal)
    defect_n = len(mobile_defect)

    y_true = np.concatenate(
        [
            np.zeros(normal_n),
            np.ones(defect_n),
        ]
    )

    mobile_scores = np.concatenate(
        [
            mobile_normal,
            mobile_defect,
        ]
    )

    fusion_scores = np.concatenate(
        [
            (
                MOBILE_WEIGHT * mobile_normal
                +
                DINO_WEIGHT * dino_normal
            ),
            (
                MOBILE_WEIGHT * mobile_defect
                +
                DINO_WEIGHT * dino_defect
            ),
        ]
    )

    observed_mobile = roc_auc_score(
        y_true,
        mobile_scores,
    )

    observed_fusion = roc_auc_score(
        y_true,
        fusion_scores,
    )

    observed_difference = (
        observed_fusion
        -
        observed_mobile
    )

    differences = []

    for _ in range(N_BOOTSTRAPS):

        normal_indices = rng.integers(
            0,
            normal_n,
            size=normal_n,
        )

        defect_indices = rng.integers(
            0,
            defect_n,
            size=defect_n,
        )

        boot_mobile = np.concatenate(
            [
                mobile_normal[
                    normal_indices
                ],
                mobile_defect[
                    defect_indices
                ],
            ]
        )

        boot_fusion = np.concatenate(
            [
                (
                    MOBILE_WEIGHT
                    * mobile_normal[
                        normal_indices
                    ]
                    +
                    DINO_WEIGHT
                    * dino_normal[
                        normal_indices
                    ]
                ),
                (
                    MOBILE_WEIGHT
                    * mobile_defect[
                        defect_indices
                    ]
                    +
                    DINO_WEIGHT
                    * dino_defect[
                        defect_indices
                    ]
                ),
            ]
        )

        boot_y = np.concatenate(
            [
                np.zeros(normal_n),
                np.ones(defect_n),
            ]
        )

        mobile_auc = roc_auc_score(
            boot_y,
            boot_mobile,
        )

        fusion_auc = roc_auc_score(
            boot_y,
            boot_fusion,
        )

        differences.append(
            fusion_auc - mobile_auc
        )

    differences = np.asarray(
        differences,
        dtype=np.float64,
    )

    ci_low, ci_high = np.percentile(
        differences,
        [2.5, 97.5],
    )

    probability_fusion_better = float(
        np.mean(differences > 0)
    )

    return {
        "mobile_auroc": float(
            observed_mobile
        ),
        "fusion_auroc": float(
            observed_fusion
        ),
        "observed_difference": float(
            observed_difference
        ),
        "bootstrap_mean_difference": float(
            np.mean(differences)
        ),
        "ci_95_low": float(
            ci_low
        ),
        "ci_95_high": float(
            ci_high
        ),
        "probability_fusion_better": (
            probability_fusion_better
        ),
    }


def main():

    print("=" * 72)
    print(
        "VISYN — FUSION BOOTSTRAP STABILITY"
    )
    print("=" * 72)

    print()
    print(
        f"Fusion: "
        f"MobileNet={MOBILE_WEIGHT:.1f}, "
        f"DINOv2={DINO_WEIGHT:.1f}"
    )

    print(
        f"Bootstrap samples: {N_BOOTSTRAPS}"
    )

    print(
        "Data: development only"
    )

    print(
        "Locked final test: NOT USED"
    )

    dino = load_json(
        DINO_PATH
    )

    mobile = load_json(
        MOBILE_PATH
    )

    # Make sure the selected fusion actually
    # exists in the saved global search.
    fusion_results = load_json(
        FUSION_PATH
    )

    selected = (
        fusion_results["max"]["best_global"]
    )

    if (
        abs(
            selected["mobile_weight"]
            - MOBILE_WEIGHT
        ) > 1e-9
        or
        abs(
            selected["dino_weight"]
            - DINO_WEIGHT
        ) > 1e-9
    ):
        raise RuntimeError(
            "Saved development fusion selection "
            "does not match the bootstrap configuration."
        )

    rng = np.random.default_rng(
        SEED
    )

    categories = list(
        dino.keys()
    )

    results = {}

    all_mobile = []
    all_fusion = []
    all_y = []

    for category in categories:

        (
            mobile_normal,
            mobile_defect,
            dino_normal,
            dino_defect,
        ) = prepare_category(
            dino,
            mobile,
            category,
        )

        result = bootstrap_category(
            mobile_normal,
            mobile_defect,
            dino_normal,
            dino_defect,
            rng,
        )

        results[category] = result

        all_mobile.extend(
            np.concatenate(
                [
                    mobile_normal,
                    mobile_defect,
                ]
            )
        )

        all_fusion.extend(
            np.concatenate(
                [
                    (
                        MOBILE_WEIGHT
                        * mobile_normal
                        +
                        DINO_WEIGHT
                        * dino_normal
                    ),
                    (
                        MOBILE_WEIGHT
                        * mobile_defect
                        +
                        DINO_WEIGHT
                        * dino_defect
                    ),
                ]
            )
        )

        all_y.extend(
            np.concatenate(
                [
                    np.zeros(
                        len(mobile_normal)
                    ),
                    np.ones(
                        len(mobile_defect)
                    ),
                ]
            )
        )

        print()
        print(
            f"{category:10s} | "
            f"MobileNet={result['mobile_auroc']:.4f} | "
            f"Fusion={result['fusion_auroc']:.4f} | "
            f"Diff={result['observed_difference']:+.4f} | "
            f"95% CI="
            f"[{result['ci_95_low']:+.4f}, "
            f"{result['ci_95_high']:+.4f}] | "
            f"P(fusion>mobile)="
            f"{result['probability_fusion_better']:.3f}"
        )

    # -------------------------------------------------------------
    # Pooled analysis
    # -------------------------------------------------------------

    all_mobile = np.asarray(
        all_mobile,
        dtype=np.float64,
    )

    all_fusion = np.asarray(
        all_fusion,
        dtype=np.float64,
    )

    all_y = np.asarray(
        all_y,
        dtype=np.float64,
    )

    pooled_mobile = roc_auc_score(
        all_y,
        all_mobile,
    )

    pooled_fusion = roc_auc_score(
        all_y,
        all_fusion,
    )

    pooled_difference = (
        pooled_fusion
        -
        pooled_mobile
    )

    print()
    print("=" * 72)
    print("POOLED DEVELOPMENT RESULT")
    print("=" * 72)

    print(
        f"MobileNet : {pooled_mobile:.4f}"
    )

    print(
        f"Fusion    : {pooled_fusion:.4f}"
    )

    print(
        f"Difference: {pooled_difference:+.4f}"
    )

    output = {
        "configuration": {
            "mobile_weight": MOBILE_WEIGHT,
            "dino_weight": DINO_WEIGHT,
            "aggregation": "max",
            "bootstrap_samples": N_BOOTSTRAPS,
            "seed": SEED,
            "data_split": "development_only",
            "final_test_used": False,
        },
        "categories": results,
        "pooled": {
            "mobile_auroc": float(
                pooled_mobile
            ),
            "fusion_auroc": float(
                pooled_fusion
            ),
            "difference": float(
                pooled_difference
            ),
        },
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

    print()
    print("=" * 72)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()