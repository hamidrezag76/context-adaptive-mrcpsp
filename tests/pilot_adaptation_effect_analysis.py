from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd
from scipy.stats import pearsonr, spearmanr


INPUT = Path(
    "results/context_analysis_j30/context_trajectory.csv"
)

OUTPUT_DIR = Path(
    "results/context_analysis_j30"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "adaptation_effect_analysis.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "adaptation_effect_summary.json"
)


CONTEXT_COLUMNS = [
    "carbon_pressure",
    "energy_pressure",
    "resource_pressure",
    "cost_pressure",
    "schedule_pressure",
    "uncertainty",
]

OPERATOR_COLUMNS = [
    "crossover_probability",
    "mutation_probability",
]

OBJECTIVE_COLUMNS = [
    "best_makespan",
    "best_cost",
    "best_carbon",
    "best_energy",
]


def safe_spearman(x, y):
    if len(x) < 3:
        return None

    if x.nunique() <= 1 or y.nunique() <= 1:
        return None

    value = spearmanr(x, y).statistic

    if not math.isfinite(float(value)):
        return None

    return float(value)


def safe_pearson(x, y):
    if len(x) < 3:
        return None

    if x.nunique() <= 1 or y.nunique() <= 1:
        return None

    value = pearsonr(x, y).statistic

    if not math.isfinite(float(value)):
        return None

    return float(value)


def main():

    print()
    print("=" * 80)
    print("CA-NSGA-II ADAPTATION EFFECT ANALYSIS")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing trajectory file: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    required = (
        ["instance", "seed", "generation"]
        + CONTEXT_COLUMNS
        + OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(
        "Input validation: PASS"
    )

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    df = df.sort_values(
        [
            "instance",
            "seed",
            "generation",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Compute generation-to-generation changes
    # ---------------------------------------------------------

    group = df.groupby(
        ["instance", "seed"],
        group_keys=False,
    )

    for column in (
        CONTEXT_COLUMNS
        + OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    ):

        df[
            f"delta_{column}"
        ] = group[column].diff()

    print(
        "Delta calculation: PASS"
    )

    # ---------------------------------------------------------
    # Remove generation 0
    # ---------------------------------------------------------

    delta_df = df[
        df["generation"] > 0
    ].copy()

    print(
        f"Transition records: {len(delta_df)}"
    )

    expected = 10 * 10 * 20

    if len(delta_df) != expected:
        raise AssertionError(
            f"Expected {expected} transitions, "
            f"got {len(delta_df)}"
        )

    print(
        "Transition count: PASS"
    )

    # ---------------------------------------------------------
    # Context -> operator delta correlations
    # ---------------------------------------------------------

    correlation_records = []

    for context in CONTEXT_COLUMNS:

        for operator in OPERATOR_COLUMNS:

            x = delta_df[
                f"delta_{context}"
            ]

            y = delta_df[
                f"delta_{operator}"
            ]

            pearson = safe_pearson(
                x,
                y,
            )

            spearman = safe_spearman(
                x,
                y,
            )

            correlation_records.append(
                {
                    "relationship": (
                        "context_to_operator"
                    ),
                    "context": context,
                    "target": operator,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )

    # ---------------------------------------------------------
    # Operator -> objective delta correlations
    # ---------------------------------------------------------

    for operator in OPERATOR_COLUMNS:

        for objective in OBJECTIVE_COLUMNS:

            x = delta_df[
                f"delta_{operator}"
            ]

            y = delta_df[
                f"delta_{objective}"
            ]

            pearson = safe_pearson(
                x,
                y,
            )

            spearman = safe_spearman(
                x,
                y,
            )

            correlation_records.append(
                {
                    "relationship": (
                        "operator_to_objective"
                    ),
                    "context": operator,
                    "target": objective,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )

    # ---------------------------------------------------------
    # Context -> objective delta correlations
    # ---------------------------------------------------------

    for context in CONTEXT_COLUMNS:

        for objective in OBJECTIVE_COLUMNS:

            x = delta_df[
                f"delta_{context}"
            ]

            y = delta_df[
                f"delta_{objective}"
            ]

            pearson = safe_pearson(
                x,
                y,
            )

            spearman = safe_spearman(
                x,
                y,
            )

            correlation_records.append(
                {
                    "relationship": (
                        "context_to_objective"
                    ),
                    "context": context,
                    "target": objective,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )

    correlation_df = pd.DataFrame(
        correlation_records
    )

    correlation_path = (
        OUTPUT_DIR
        / "adaptation_delta_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    print(
        f"Saved: {correlation_path}"
    )

    # ---------------------------------------------------------
    # Strong relationships
    # ---------------------------------------------------------

    strong = correlation_df[
        correlation_df["spearman"].abs()
        >= 0.30
    ].copy()

    strong = strong.sort_values(
        "spearman",
        key=lambda s: s.abs(),
        ascending=False,
    )

    print()
    print("=" * 80)
    print(
        "STRONG DELTA RELATIONSHIPS | |rho| >= 0.30"
    )
    print("=" * 80)

    for _, row in strong.iterrows():

        print(
            f"{row['context']} -> "
            f"{row['target']}: "
            f"Spearman="
            f"{row['spearman']:+.4f}"
        )

    # ---------------------------------------------------------
    # Initial -> final adaptation by run
    # ---------------------------------------------------------

    run_records = []

    for (
        instance,
        seed,
    ), run_df in df.groupby(
        ["instance", "seed"]
    ):

        run_df = run_df.sort_values(
            "generation"
        )

        first = run_df.iloc[0]
        last = run_df.iloc[-1]

        record = {
            "instance": instance,
            "seed": int(seed),
        }

        for column in (
            CONTEXT_COLUMNS
            + OPERATOR_COLUMNS
            + OBJECTIVE_COLUMNS
        ):

            record[
                f"initial_{column}"
            ] = float(first[column])

            record[
                f"final_{column}"
            ] = float(last[column])

            record[
                f"delta_{column}"
            ] = float(
                last[column]
                - first[column]
            )

        run_records.append(
            record
        )

    run_df = pd.DataFrame(
        run_records
    )

    run_path = (
        OUTPUT_DIR
        / "adaptation_run_profiles.csv"
    )

    run_df.to_csv(
        run_path,
        index=False,
    )

    print()
    print(
        f"Saved: {run_path}"
    )

    # ---------------------------------------------------------
    # Aggregate adaptation
    # ---------------------------------------------------------

    aggregate = {}

    for column in (
        CONTEXT_COLUMNS
        + OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    ):

        values = run_df[
            f"delta_{column}"
        ]

        aggregate[column] = {
            "mean_delta": float(
                values.mean()
            ),
            "median_delta": float(
                values.median()
            ),
            "std_delta": float(
                values.std()
            ),
            "min_delta": float(
                values.min()
            ),
            "max_delta": float(
                values.max()
            ),
        }

    # ---------------------------------------------------------
    # Operator adaptation summary
    # ---------------------------------------------------------

    pc_delta = run_df[
        "delta_crossover_probability"
    ]

    pm_delta = run_df[
        "delta_mutation_probability"
    ]

    operator_summary = {
        "crossover_probability": {
            "mean_delta": float(
                pc_delta.mean()
            ),
            "median_delta": float(
                pc_delta.median()
            ),
            "runs_decreased": int(
                (pc_delta < 0).sum()
            ),
            "runs_increased": int(
                (pc_delta > 0).sum()
            ),
            "runs_unchanged": int(
                (pc_delta == 0).sum()
            ),
        },
        "mutation_probability": {
            "mean_delta": float(
                pm_delta.mean()
            ),
            "median_delta": float(
                pm_delta.median()
            ),
            "runs_decreased": int(
                (pm_delta < 0).sum()
            ),
            "runs_increased": int(
                (pm_delta > 0).sum()
            ),
            "runs_unchanged": int(
                (pm_delta == 0).sum()
            ),
        },
    }

    # ---------------------------------------------------------
    # Context pressure summary
    # ---------------------------------------------------------

    context_summary = {}

    for column in CONTEXT_COLUMNS:

        values = run_df[
            f"delta_{column}"
        ]

        context_summary[column] = {
            "mean_delta": float(
                values.mean()
            ),
            "median_delta": float(
                values.median()
            ),
            "runs_decreased": int(
                (values < 0).sum()
            ),
            "runs_increased": int(
                (values > 0).sum()
            ),
        }

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "transition_records": int(
                len(delta_df)
            ),
            "instances": int(
                df["instance"].nunique()
            ),
            "seeds": int(
                df["seed"].nunique()
            ),
            "generations": int(
                df["generation"].nunique()
            ),
        },
        "operator_adaptation":
            operator_summary,
        "context_changes":
            context_summary,
        "aggregate_changes":
            aggregate,
        "strong_relationship_count":
            int(len(strong)),
        "strong_relationships":
            strong.to_dict(
                orient="records"
            ),
    }

    with OUTPUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "OPERATOR ADAPTATION SUMMARY"
    )
    print("=" * 80)

    print(
        "Mean ΔPc:",
        f"{pc_delta.mean():+.6f}",
    )

    print(
        "Mean ΔPm:",
        f"{pm_delta.mean():+.6f}",
    )

    print(
        "Runs with decreased Pc:",
        int((pc_delta < 0).sum()),
        "/",
        len(pc_delta),
    )

    print(
        "Runs with increased Pm:",
        int((pm_delta > 0).sum()),
        "/",
        len(pm_delta),
    )

    print()
    print(
        "CA-NSGA-II ADAPTATION EFFECT ANALYSIS: COMPLETE"
    )


if __name__ == "__main__":
    main()