"""
context_operator_analysis.py

Post-hoc analysis of the real CA-NSGA-II context trajectory.

Input:
    results/context_analysis_j30/context_trajectory.csv

The script analyzes:

1. Context -> crossover probability
2. Context -> mutation probability
3. Context -> optimization trajectory
4. Context variability
5. Per-instance context profiles
6. Per-instance operator profiles

No new optimization run is performed.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "results/context_analysis_j30/context_trajectory.csv"
)

OUTPUT_ROOT = Path(
    "results/context_analysis_j30"
)

CORRELATION_FILE = (
    OUTPUT_ROOT
    / "context_operator_correlations.csv"
)

INSTANCE_FILE = (
    OUTPUT_ROOT
    / "context_instance_profiles.csv"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "context_operator_analysis.json"
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


# ============================================================
# Validation
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print("=" * 80)
    print("INPUT VALIDATION")
    print("=" * 80)

    print(
        "Rows:",
        len(df),
    )

    print(
        "Columns:",
        len(df.columns),
    )

    required = (
        ["instance", "seed", "generation"]
        + CONTEXT_COLUMNS
        + OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    assert len(df) == 2100

    assert (
        df["instance"].nunique()
        == 10
    )

    assert (
        df["seed"].nunique()
        == 10
    )

    assert (
        df["generation"].nunique()
        == 21
    )

    print(
        "Required columns: PASS"
    )

    print(
        "Dataset dimensions: PASS"
    )

    return df


# ============================================================
# Pearson correlations
# ============================================================

def correlation_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    targets = (
        OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    )

    for context_column in CONTEXT_COLUMNS:

        for target in targets:

            x = df[
                context_column
            ].astype(float)

            y = df[
                target
            ].astype(float)

            pearson = x.corr(
                y,
                method="pearson",
            )

            spearman = x.corr(
                y,
                method="spearman",
            )

            rows.append(
                {
                    "context_variable":
                        context_column,

                    "target_variable":
                        target,

                    "pearson_r":
                        float(pearson),

                    "spearman_rho":
                        float(spearman),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        CORRELATION_FILE,
        index=False,
    )

    return result


# ============================================================
# Per-instance profiles
# ============================================================

def instance_profiles(
    df: pd.DataFrame,
) -> pd.DataFrame:

    aggregations = {}

    for column in (
        CONTEXT_COLUMNS
        + OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    ):

        aggregations[
            f"{column}_mean"
        ] = (
            column,
            "mean",
        )

        aggregations[
            f"{column}_std"
        ] = (
            column,
            "std",
        )

        aggregations[
            f"{column}_min"
        ] = (
            column,
            "min",
        )

        aggregations[
            f"{column}_max"
        ] = (
            column,
            "max",
        )

    result = (
        df.groupby(
            "instance"
        )
        .agg(
            **{
                name: pd.NamedAgg(
                    column=column,
                    aggfunc=aggfunc,
                )
                for name, (
                    column,
                    aggfunc,
                )
                in aggregations.items()
            }
        )
        .reset_index()
    )

    result.to_csv(
        INSTANCE_FILE,
        index=False,
    )

    return result


# ============================================================
# Context variability
# ============================================================

def context_variability(
    df: pd.DataFrame,
) -> dict:

    result = {}

    for column in CONTEXT_COLUMNS:

        values = (
            df[column]
            .astype(float)
        )

        result[column] = {
            "mean":
                float(values.mean()),

            "std":
                float(values.std()),

            "min":
                float(values.min()),

            "max":
                float(values.max()),

            "range":
                float(
                    values.max()
                    - values.min()
                ),
        }

    return result


# ============================================================
# Operator variability
# ============================================================

def operator_variability(
    df: pd.DataFrame,
) -> dict:

    result = {}

    for column in OPERATOR_COLUMNS:

        values = (
            df[column]
            .astype(float)
        )

        result[column] = {
            "mean":
                float(values.mean()),

            "std":
                float(values.std()),

            "min":
                float(values.min()),

            "max":
                float(values.max()),

            "range":
                float(
                    values.max()
                    - values.min()
                ),
        }

    return result


# ============================================================
# Generation-level dynamics
# ============================================================

def generation_profiles(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = (
        CONTEXT_COLUMNS
        + OPERATOR_COLUMNS
        + OBJECTIVE_COLUMNS
    )

    result = (
        df.groupby(
            "generation"
        )[columns]
        .mean()
        .reset_index()
    )

    path = (
        OUTPUT_ROOT
        / "context_generation_profiles.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    return result


# ============================================================
# Context -> operator interpretation
# ============================================================

def operator_relationships(
    correlation_df: pd.DataFrame,
) -> dict:

    result = {}

    for operator in OPERATOR_COLUMNS:

        subset = correlation_df[
            correlation_df[
                "target_variable"
            ]
            == operator
        ]

        result[operator] = {}

        for _, row in subset.iterrows():

            result[operator][
                row["context_variable"]
            ] = {
                "pearson_r":
                    float(
                        row["pearson_r"]
                    ),

                "spearman_rho":
                    float(
                        row["spearman_rho"]
                    ),
            }

    return result


# ============================================================
# Context -> objective interpretation
# ============================================================

def objective_relationships(
    correlation_df: pd.DataFrame,
) -> dict:

    result = {}

    for objective in OBJECTIVE_COLUMNS:

        subset = correlation_df[
            correlation_df[
                "target_variable"
            ]
            == objective
        ]

        result[objective] = {}

        for _, row in subset.iterrows():

            result[objective][
                row["context_variable"]
            ] = {
                "pearson_r":
                    float(
                        row["pearson_r"]
                    ),

                "spearman_rho":
                    float(
                        row["spearman_rho"]
                    ),
            }

    return result


# ============================================================
# Strong relationships
# ============================================================

def strongest_relationships(
    correlation_df: pd.DataFrame,
    threshold: float = 0.30,
) -> list[dict]:

    result = []

    for _, row in correlation_df.iterrows():

        pearson = float(
            row["pearson_r"]
        )

        spearman = float(
            row["spearman_rho"]
        )

        strength = max(
            abs(pearson),
            abs(spearman),
        )

        if strength >= threshold:

            result.append(
                {
                    "context_variable":
                        row["context_variable"],

                    "target_variable":
                        row["target_variable"],

                    "pearson_r":
                        pearson,

                    "spearman_rho":
                        spearman,

                    "max_absolute_correlation":
                        strength,
                }
            )

    result.sort(
        key=lambda x:
            x["max_absolute_correlation"],
        reverse=True,
    )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CONTEXT–OPERATOR ANALYSIS")
    print("=" * 80)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    correlations = correlation_analysis(
        df
    )

    print(
        "Correlation matrix records:",
        len(correlations),
    )

    print(
        "Saved:",
        CORRELATION_FILE,
    )

    # --------------------------------------------------------
    # Instance profiles
    # --------------------------------------------------------

    profiles = instance_profiles(
        df
    )

    print()
    print(
        "Instance profiles:",
        len(profiles),
    )

    print(
        "Saved:",
        INSTANCE_FILE,
    )

    # --------------------------------------------------------
    # Generation profiles
    # --------------------------------------------------------

    generation = generation_profiles(
        df
    )

    print()
    print(
        "Generation profiles:",
        len(generation),
    )

    # --------------------------------------------------------
    # Variability
    # --------------------------------------------------------

    context_stats = context_variability(
        df
    )

    operator_stats = operator_variability(
        df
    )

    # --------------------------------------------------------
    # Relationships
    # --------------------------------------------------------

    operator_relations = (
        operator_relationships(
            correlations
        )
    )

    objective_relations = (
        objective_relationships(
            correlations
        )
    )

    strong = strongest_relationships(
        correlations
    )

    # --------------------------------------------------------
    # Print key operator correlations
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CONTEXT → OPERATOR RELATIONSHIPS")
    print("=" * 80)

    for operator in OPERATOR_COLUMNS:

        print()
        print(operator)

        subset = correlations[
            correlations[
                "target_variable"
            ]
            == operator
        ]

        subset = subset.sort_values(
            "spearman_rho",
            ascending=False,
        )

        for _, row in subset.iterrows():

            print(
                f"  {row['context_variable']}: "
                f"Pearson={row['pearson_r']:+.4f}, "
                f"Spearman={row['spearman_rho']:+.4f}"
            )

    # --------------------------------------------------------
    # Print key objective correlations
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CONTEXT → OBJECTIVE RELATIONSHIPS")
    print("=" * 80)

    for objective in OBJECTIVE_COLUMNS:

        print()
        print(objective)

        subset = correlations[
            correlations[
                "target_variable"
            ]
            == objective
        ]

        subset = subset.sort_values(
            "spearman_rho",
            ascending=False,
        )

        for _, row in subset.iterrows():

            print(
                f"  {row['context_variable']}: "
                f"Pearson={row['pearson_r']:+.4f}, "
                f"Spearman={row['spearman_rho']:+.4f}"
            )

    # --------------------------------------------------------
    # Strong relationships
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STRONGER RELATIONSHIPS | |rho| >= 0.30")
    print("=" * 80)

    if not strong:

        print(
            "No relationships exceeded the threshold."
        )

    else:

        for item in strong:

            print(
                f"{item['context_variable']} "
                f"→ "
                f"{item['target_variable']}: "
                f"Spearman={item['spearman_rho']:+.4f}"
            )

    # --------------------------------------------------------
    # Save complete JSON
    # --------------------------------------------------------

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "instances":
                int(df["instance"].nunique()),
            "seeds":
                int(df["seed"].nunique()),
            "generations":
                int(df["generation"].nunique()),
        },

        "context_variables":
            CONTEXT_COLUMNS,

        "operator_variables":
            OPERATOR_COLUMNS,

        "objective_variables":
            OBJECTIVE_COLUMNS,

        "context_variability":
            context_stats,

        "operator_variability":
            operator_stats,

        "context_to_operator":
            operator_relations,

        "context_to_objective":
            objective_relations,

        "strong_relationships":
            strong,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    assert len(df) == 2100

    assert len(correlations) == (
        len(CONTEXT_COLUMNS)
        * (
            len(OPERATOR_COLUMNS)
            + len(OBJECTIVE_COLUMNS)
        )
    )

    assert len(profiles) == 10

    assert len(generation) == 21

    assert SUMMARY_FILE.exists()

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    print(
        "Dataset: PASS"
    )

    print(
        "Correlation analysis: PASS"
    )

    print(
        "Instance profiles: PASS"
    )

    print(
        "Generation profiles: PASS"
    )

    print(
        "Summary persistence: PASS"
    )

    print()
    print("=" * 80)
    print("CONTEXT–OPERATOR ANALYSIS: COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()