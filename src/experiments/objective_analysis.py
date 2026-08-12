"""
objective_analysis.py

Cross-instance statistical analysis of the four
construction scheduling objectives.

Objectives:
    0 = Makespan
    1 = Cost
    2 = Carbon
    3 = Energy

Data source:
    results/pilot_j30/

Experimental design:
    10 PSPLIB J30 instances
    10 seeds
    2 algorithms

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev

from scipy.stats import wilcoxon, ttest_rel


# =========================================================
# Configuration
# =========================================================

RESULTS_ROOT = Path(
    "results/pilot_j30"
)

INSTANCES = [
    f"j3010_{i}.mm"
    for i in range(1, 11)
]

SEEDS = list(
    range(42, 52)
)

BASELINE = "baseline_nsga2"
ADAPTIVE = "ca_nsga2"

OBJECTIVES = {
    "makespan": 0,
    "cost": 1,
    "carbon": 2,
    "energy": 3,
}

ALPHA = 0.05


# =========================================================
# Load
# =========================================================

def load_record(
    instance: str,
    algorithm: str,
    seed: int,
) -> dict:

    path = (
        RESULTS_ROOT
        / Path(instance).stem
        / algorithm
        / f"seed_{seed}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        record = json.load(file)

    if record["instance"] != instance:
        raise ValueError(
            f"Instance mismatch: {path}"
        )

    if record["algorithm"] != algorithm:
        raise ValueError(
            f"Algorithm mismatch: {path}"
        )

    if int(record["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch: {path}"
        )

    archive = record.get(
        "archive_objectives"
    )

    if not archive:
        raise ValueError(
            f"Empty archive: {path}"
        )

    for point in archive:

        if len(point) != 4:
            raise ValueError(
                f"Expected 4 objectives: {path}"
            )

    return record


# =========================================================
# Extract objective-specific best values
# =========================================================

def objective_best(
    record: dict,
    objective_index: int,
) -> float:

    archive = record[
        "archive_objectives"
    ]

    values = [
        float(point[objective_index])
        for point in archive
    ]

    return min(values)


# =========================================================
# Load all data
# =========================================================

def load_data():

    data = {}

    for instance in INSTANCES:

        data[instance] = {}

        for seed in SEEDS:

            data[instance][seed] = {}

            for algorithm in [
                BASELINE,
                ADAPTIVE,
            ]:

                data[
                    instance
                ][seed][algorithm] = (
                    load_record(
                        instance,
                        algorithm,
                        seed,
                    )
                )

    return data


# =========================================================
# Statistics
# =========================================================

def cohens_d(
    baseline: list[float],
    adaptive: list[float],
) -> float:

    differences = [
        a - b
        for b, a in zip(
            baseline,
            adaptive,
        )
    ]

    if len(differences) < 2:
        return 0.0

    sd = stdev(
        differences
    )

    if sd == 0.0:

        if mean(differences) == 0.0:
            return 0.0

        return float("inf")

    return (
        mean(differences)
        / sd
    )


def ci95(
    differences: list[float],
) -> tuple[float, float]:

    m = mean(
        differences
    )

    if len(differences) < 2:
        return m, m

    sd = stdev(
        differences
    )

    se = (
        sd
        / sqrt(len(differences))
    )

    margin = (
        1.96
        * se
    )

    return (
        m - margin,
        m + margin,
    )


def compare(
    baseline: list[float],
    adaptive: list[float],
) -> dict:

    differences = [
        a - b
        for b, a in zip(
            baseline,
            adaptive,
        )
    ]

    b_mean = mean(
        baseline
    )

    a_mean = mean(
        adaptive
    )

    wins_ca = sum(
        a < b
        for b, a in zip(
            baseline,
            adaptive,
        )
    )

    wins_baseline = sum(
        b < a
        for b, a in zip(
            baseline,
            adaptive,
        )
    )

    ties = sum(
        a == b
        for b, a in zip(
            baseline,
            adaptive,
        )
    )

    try:

        w = wilcoxon(
            baseline,
            adaptive,
            alternative="two-sided",
            zero_method="wilcox",
        )

        w_stat = float(
            w.statistic
        )

        w_p = float(
            w.pvalue
        )

    except ValueError:

        w_stat = 0.0
        w_p = 1.0

    t = ttest_rel(
        adaptive,
        baseline,
    )

    relative_improvement = (
        (
            b_mean - a_mean
        )
        / abs(b_mean)
    ) * 100.0

    low, high = ci95(
        differences
    )

    return {

        "n": len(baseline),

        "baseline_mean": b_mean,

        "baseline_sd": stdev(
            baseline
        ),

        "adaptive_mean": a_mean,

        "adaptive_sd": stdev(
            adaptive
        ),

        "mean_difference":
            mean(differences),

        "median_difference":
            median(differences),

        "ci95_low": low,

        "ci95_high": high,

        "ca_wins": wins_ca,

        "baseline_wins":
            wins_baseline,

        "ties": ties,

        "wilcoxon_statistic":
            w_stat,

        "wilcoxon_p":
            w_p,

        "paired_t_statistic":
            float(t.statistic),

        "paired_t_p":
            float(t.pvalue),

        "cohens_d":
            cohens_d(
                baseline,
                adaptive,
            ),

        "relative_improvement":
            relative_improvement,

        "significant":
            w_p < ALPHA,
    }


# =========================================================
# Seed-level extraction
# =========================================================

def seed_level(
    data,
    objective_index: int,
):

    baseline = []
    adaptive = []

    for instance in INSTANCES:

        for seed in SEEDS:

            b = objective_best(
                data[instance][seed][BASELINE],
                objective_index,
            )

            a = objective_best(
                data[instance][seed][ADAPTIVE],
                objective_index,
            )

            baseline.append(b)
            adaptive.append(a)

    return (
        baseline,
        adaptive,
    )


# =========================================================
# Instance-level extraction
# =========================================================

def instance_level(
    data,
    objective_index: int,
):

    baseline = []
    adaptive = []

    rows = []

    for instance in INSTANCES:

        b_values = []
        a_values = []

        for seed in SEEDS:

            b_values.append(
                objective_best(
                    data[instance][seed][BASELINE],
                    objective_index,
                )
            )

            a_values.append(
                objective_best(
                    data[instance][seed][ADAPTIVE],
                    objective_index,
                )
            )

        b_mean = mean(
            b_values
        )

        a_mean = mean(
            a_values
        )

        baseline.append(
            b_mean
        )

        adaptive.append(
            a_mean
        )

        rows.append(
            (
                instance,
                b_mean,
                a_mean,
            )
        )

    return (
        baseline,
        adaptive,
        rows,
    )


# =========================================================
# Printing
# =========================================================

def print_result(
    objective: str,
    level: str,
    result: dict,
):

    print()
    print(
        "=" * 60
    )

    print(
        f"{objective.upper()} — {level}"
    )

    print(
        "=" * 60
    )

    print(
        f"N: {result['n']}"
    )

    print(
        f"Baseline mean: "
        f"{result['baseline_mean']:.6f}"
    )

    print(
        f"Baseline SD: "
        f"{result['baseline_sd']:.6f}"
    )

    print(
        f"CA mean: "
        f"{result['adaptive_mean']:.6f}"
    )

    print(
        f"CA SD: "
        f"{result['adaptive_sd']:.6f}"
    )

    print(
        f"Mean difference: "
        f"{result['mean_difference']:.6f}"
    )

    print(
        f"Median difference: "
        f"{result['median_difference']:.6f}"
    )

    print(
        f"95% CI: "
        f"[{result['ci95_low']:.6f}, "
        f"{result['ci95_high']:.6f}]"
    )

    print(
        f"CA wins: "
        f"{result['ca_wins']}"
    )

    print(
        f"Baseline wins: "
        f"{result['baseline_wins']}"
    )

    print(
        f"Ties: "
        f"{result['ties']}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{result['wilcoxon_statistic']:.6f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{result['wilcoxon_p']:.8f}"
    )

    print(
        f"Paired t statistic: "
        f"{result['paired_t_statistic']:.6f}"
    )

    print(
        f"Paired t p-value: "
        f"{result['paired_t_p']:.8f}"
    )

    print(
        f"Cohen's d: "
        f"{result['cohens_d']:.6f}"
    )

    print(
        f"Relative improvement: "
        f"{result['relative_improvement']:.4f}%"
    )

    print(
        f"Significant at α=0.05: "
        f"{result['significant']}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print()
    print(
        "=" * 60
    )
    print(
        "FOUR-OBJECTIVE CROSS-INSTANCE ANALYSIS"
    )
    print(
        "=" * 60
    )

    print(
        f"Instances: {len(INSTANCES)}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Total paired runs: "
        f"{len(INSTANCES) * len(SEEDS)}"
    )

    data = load_data()

    print(
        "\nData loading: PASS"
    )

    # -----------------------------------------------------
    # Seed-level
    # -----------------------------------------------------

    print()
    print(
        "################################################"
    )
    print(
        "SEED-LEVEL ANALYSIS"
    )
    print(
        "################################################"
    )

    for objective, index in OBJECTIVES.items():

        baseline, adaptive = (
            seed_level(
                data,
                index,
            )
        )

        result = compare(
            baseline,
            adaptive,
        )

        print_result(
            objective,
            "SEED LEVEL",
            result,
        )

    # -----------------------------------------------------
    # Instance-level
    # -----------------------------------------------------

    print()
    print(
        "################################################"
    )
    print(
        "INSTANCE-LEVEL ANALYSIS"
    )
    print(
        "################################################"
    )

    instance_tables = {}

    for objective, index in OBJECTIVES.items():

        baseline, adaptive, rows = (
            instance_level(
                data,
                index,
            )
        )

        result = compare(
            baseline,
            adaptive,
        )

        instance_tables[
            objective
        ] = rows

        print_result(
            objective,
            "INSTANCE LEVEL",
            result,
        )

    # -----------------------------------------------------
    # Instance table
    # -----------------------------------------------------

    print()
    print(
        "=" * 90
    )
    print(
        "INSTANCE-LEVEL OBJECTIVE MEANS"
    )
    print(
        "=" * 90
    )

    for instance_index, instance in enumerate(
        INSTANCES
    ):

        values = []

        for objective in OBJECTIVES:

            row = instance_tables[
                objective
            ][instance_index]

            values.append(
                (
                    row[1],
                    row[2],
                )
            )

        print(
            f"{instance:15s} | "
            f"M {values[0][0]:8.2f} -> {values[0][1]:8.2f} | "
            f"C {values[1][0]:12.2f} -> {values[1][1]:12.2f} | "
            f"CO2 {values[2][0]:10.2f} -> {values[2][1]:10.2f} | "
            f"E {values[3][0]:10.2f} -> {values[3][1]:10.2f}"
        )

    print()
    print(
        "FOUR-OBJECTIVE ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()