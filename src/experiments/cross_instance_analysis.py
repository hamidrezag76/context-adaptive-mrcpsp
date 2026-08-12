"""
cross_instance_analysis.py

Cross-instance statistical analysis for the PSPLIB J30 pilot.

Uses only persisted experimental results from:
    results/pilot_j30/

Experimental design:
    10 instances
    10 seeds
    2 algorithms

Metrics:
    Hypervolume
    IGD+

The analysis is performed at two levels:

1. Seed-level:
   10 instances x 10 seeds = 100 paired observations

2. Instance-level:
   10 paired instance means

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median, stdev
from math import sqrt

from scipy.stats import (
    wilcoxon,
    ttest_rel,
)


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

ALGORITHMS = (
    "baseline_nsga2",
    "ca_nsga2",
)

ALPHA = 0.05


# =========================================================
# Data loading
# =========================================================

def load_record(
    instance: str,
    algorithm: str,
    seed: int,
) -> dict:

    instance_dir = Path(instance).stem

    path = (
        RESULTS_ROOT
        / instance_dir
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

    # -----------------------------------------------------
    # Structural validation
    # -----------------------------------------------------

    if record["instance"] != instance:
        raise ValueError(
            f"Instance mismatch in {path}"
        )

    if record["algorithm"] != algorithm:
        raise ValueError(
            f"Algorithm mismatch in {path}"
        )

    if int(record["seed"]) != seed:
        raise ValueError(
            f"Seed mismatch in {path}"
        )

    metrics = record.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(
            f"Missing metrics in {path}"
        )

    if "hypervolume" not in metrics:
        raise ValueError(
            f"Missing hypervolume in {path}"
        )

    if "igd_plus" not in metrics:
        raise ValueError(
            f"Missing IGD+ in {path}"
        )

    metadata = record.get(
        "metadata",
        {},
    )

    if metadata.get(
        "metric_reference_set"
    ) != "common_all_seeds":

        raise ValueError(
            f"Invalid reference-set protocol in {path}"
        )

    return record


# =========================================================
# Load complete experiment
# =========================================================

def load_all_results():

    data = {}

    for instance in INSTANCES:

        data[instance] = {}

        for seed in SEEDS:

            data[instance][seed] = {}

            for algorithm in ALGORITHMS:

                data[instance][seed][algorithm] = (
                    load_record(
                        instance,
                        algorithm,
                        seed,
                    )
                )

    return data


# =========================================================
# Effect-size calculations
# =========================================================

def paired_cohens_d(
    baseline: list[float],
    adaptive: list[float],
) -> float:

    differences = [
        adaptive_value - baseline_value
        for baseline_value, adaptive_value
        in zip(
            baseline,
            adaptive,
        )
    ]

    if len(differences) < 2:
        return 0.0

    difference_mean = mean(
        differences
    )

    difference_sd = stdev(
        differences
    )

    if difference_sd == 0.0:

        if difference_mean == 0.0:
            return 0.0

        return float("inf")

    return (
        difference_mean
        / difference_sd
    )


def confidence_interval_95(
    differences: list[float],
) -> tuple[float, float]:

    if not differences:
        raise ValueError(
            "Cannot compute CI from empty data."
        )

    difference_mean = mean(
        differences
    )

    if len(differences) == 1:
        return (
            difference_mean,
            difference_mean,
        )

    difference_sd = stdev(
        differences
    )

    standard_error = (
        difference_sd
        / sqrt(len(differences))
    )

    # Normal approximation.
    # With n=100 this is appropriate for the
    # seed-level descriptive analysis.
    margin = (
        1.96
        * standard_error
    )

    return (
        difference_mean - margin,
        difference_mean + margin,
    )


# =========================================================
# Statistical comparison
# =========================================================

def compare(
    baseline: list[float],
    adaptive: list[float],
    higher_is_better: bool,
) -> dict:

    if len(baseline) != len(adaptive):
        raise ValueError(
            "Paired samples must have equal length."
        )

    if len(baseline) < 2:
        raise ValueError(
            "At least two paired observations are required."
        )

    differences = [
        adaptive_value - baseline_value
        for baseline_value, adaptive_value
        in zip(
            baseline,
            adaptive,
        )
    ]

    baseline_mean = mean(
        baseline
    )

    adaptive_mean = mean(
        adaptive
    )

    baseline_sd = stdev(
        baseline
    )

    adaptive_sd = stdev(
        adaptive
    )

    mean_difference = mean(
        differences
    )

    median_difference = median(
        differences
    )

    adaptive_wins = 0
    baseline_wins = 0
    ties = 0

    for b, a in zip(
        baseline,
        adaptive,
    ):

        if higher_is_better:

            if a > b:
                adaptive_wins += 1
            elif b > a:
                baseline_wins += 1
            else:
                ties += 1

        else:

            if a < b:
                adaptive_wins += 1
            elif b < a:
                baseline_wins += 1
            else:
                ties += 1

    # -----------------------------------------------------
    # Wilcoxon
    # -----------------------------------------------------

    try:

        wilcoxon_result = wilcoxon(
            baseline,
            adaptive,
            alternative="two-sided",
            zero_method="wilcox",
        )

        wilcoxon_statistic = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

    except ValueError:

        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0

    # -----------------------------------------------------
    # Paired t-test
    # -----------------------------------------------------

    t_result = ttest_rel(
        adaptive,
        baseline,
    )

    t_statistic = float(
        t_result.statistic
    )

    t_p = float(
        t_result.pvalue
    )

    # -----------------------------------------------------
    # Effect size
    # -----------------------------------------------------

    cohens_d = paired_cohens_d(
        baseline,
        adaptive,
    )

    # -----------------------------------------------------
    # Relative improvement
    # -----------------------------------------------------

    if baseline_mean == 0.0:

        relative_improvement = 0.0

    elif higher_is_better:

        relative_improvement = (
            (
                adaptive_mean
                - baseline_mean
            )
            / abs(baseline_mean)
        ) * 100.0

    else:

        relative_improvement = (
            (
                baseline_mean
                - adaptive_mean
            )
            / abs(baseline_mean)
        ) * 100.0

    ci_low, ci_high = (
        confidence_interval_95(
            differences
        )
    )

    return {

        "n":
            len(baseline),

        "baseline_mean":
            baseline_mean,

        "baseline_sd":
            baseline_sd,

        "adaptive_mean":
            adaptive_mean,

        "adaptive_sd":
            adaptive_sd,

        "mean_difference":
            mean_difference,

        "median_difference":
            median_difference,

        "ci95_low":
            ci_low,

        "ci95_high":
            ci_high,

        "adaptive_wins":
            adaptive_wins,

        "baseline_wins":
            baseline_wins,

        "ties":
            ties,

        "wilcoxon_statistic":
            wilcoxon_statistic,

        "wilcoxon_p":
            wilcoxon_p,

        "paired_t_statistic":
            t_statistic,

        "paired_t_p":
            t_p,

        "cohens_d":
            cohens_d,

        "relative_improvement_percent":
            relative_improvement,

        "significant_wilcoxon":
            wilcoxon_p < ALPHA,

        "significant_ttest":
            t_p < ALPHA,
    }


# =========================================================
# Extract seed-level observations
# =========================================================

def extract_seed_level(
    data,
    metric: str,
):

    baseline = []
    adaptive = []

    rows = []

    for instance in INSTANCES:

        for seed in SEEDS:

            b = float(
                data[instance][seed][
                    "baseline_nsga2"
                ]["metrics"][metric]
            )

            a = float(
                data[instance][seed][
                    "ca_nsga2"
                ]["metrics"][metric]
            )

            baseline.append(b)
            adaptive.append(a)

            rows.append(
                {
                    "instance": instance,
                    "seed": seed,
                    "baseline": b,
                    "adaptive": a,
                }
            )

    return (
        baseline,
        adaptive,
        rows,
    )


# =========================================================
# Extract instance-level observations
# =========================================================

def extract_instance_level(
    data,
    metric: str,
):

    baseline = []
    adaptive = []

    rows = []

    for instance in INSTANCES:

        baseline_values = []
        adaptive_values = []

        for seed in SEEDS:

            baseline_values.append(
                float(
                    data[instance][seed][
                        "baseline_nsga2"
                    ]["metrics"][metric]
                )
            )

            adaptive_values.append(
                float(
                    data[instance][seed][
                        "ca_nsga2"
                    ]["metrics"][metric]
                )
            )

        b_mean = mean(
            baseline_values
        )

        a_mean = mean(
            adaptive_values
        )

        baseline.append(
            b_mean
        )

        adaptive.append(
            a_mean
        )

        rows.append(
            {
                "instance": instance,
                "baseline_mean": b_mean,
                "adaptive_mean": a_mean,
            }
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
    title: str,
    result: dict,
):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"N: {result['n']}"
    )

    print(
        f"Baseline mean: "
        f"{result['baseline_mean']:.8f}"
    )

    print(
        f"Baseline SD: "
        f"{result['baseline_sd']:.8f}"
    )

    print(
        f"CA mean: "
        f"{result['adaptive_mean']:.8f}"
    )

    print(
        f"CA SD: "
        f"{result['adaptive_sd']:.8f}"
    )

    print(
        f"Mean difference: "
        f"{result['mean_difference']:.8f}"
    )

    print(
        f"Median difference: "
        f"{result['median_difference']:.8f}"
    )

    print(
        f"95% CI of difference: "
        f"[{result['ci95_low']:.8f}, "
        f"{result['ci95_high']:.8f}]"
    )

    print(
        f"CA wins: "
        f"{result['adaptive_wins']}"
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
        f"{result['wilcoxon_statistic']:.8f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{result['wilcoxon_p']:.8f}"
    )

    print(
        f"Paired t statistic: "
        f"{result['paired_t_statistic']:.8f}"
    )

    print(
        f"Paired t p-value: "
        f"{result['paired_t_p']:.8f}"
    )

    print(
        f"Cohen's d: "
        f"{result['cohens_d']:.8f}"
    )

    print(
        f"Relative improvement: "
        f"{result['relative_improvement_percent']:.4f}%"
    )

    print(
        f"Wilcoxon significant: "
        f"{result['significant_wilcoxon']}"
    )

    print(
        f"Paired t-test significant: "
        f"{result['significant_ttest']}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 60)
    print("CROSS-INSTANCE ANALYSIS")
    print("=" * 60)

    print(
        f"Results root: {RESULTS_ROOT}"
    )

    print(
        f"Instances: {len(INSTANCES)}"
    )

    print(
        f"Seeds per instance: {len(SEEDS)}"
    )

    print(
        f"Paired seed observations: "
        f"{len(INSTANCES) * len(SEEDS)}"
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    data = load_all_results()

    print(
        "\nData loading: PASS"
    )

    # -----------------------------------------------------
    # Verify common protocol
    # -----------------------------------------------------

    reference_sets = set()
    reference_points = set()

    for instance in INSTANCES:

        for seed in SEEDS:

            for algorithm in ALGORITHMS:

                metadata = data[
                    instance
                ][seed][algorithm][
                    "metadata"
                ]

                reference_sets.add(
                    metadata[
                        "metric_reference_set"
                    ]
                )

                reference_points.add(
                    tuple(
                        metadata[
                            "metric_reference_point"
                        ]
                    )
                )

    print(
        "Reference-set protocols:",
        reference_sets,
    )

    print(
        "Reference points:",
        reference_points,
    )

    assert reference_sets == {
        "common_all_seeds"
    }

    assert len(reference_points) == 1

    print(
        "Common metric protocol: PASS"
    )

    # =====================================================
    # Seed-level analysis
    # =====================================================

    for metric, higher_is_better in [
        ("hypervolume", True),
        ("igd_plus", False),
    ]:

        (
            baseline,
            adaptive,
            rows,
        ) = extract_seed_level(
            data,
            metric,
        )

        result = compare(
            baseline,
            adaptive,
            higher_is_better,
        )

        print_result(
            f"{metric.upper()} — SEED LEVEL",
            result,
        )

    # =====================================================
    # Instance-level analysis
    # =====================================================

    for metric, higher_is_better in [
        ("hypervolume", True),
        ("igd_plus", False),
    ]:

        (
            baseline,
            adaptive,
            rows,
        ) = extract_instance_level(
            data,
            metric,
        )

        result = compare(
            baseline,
            adaptive,
            higher_is_better,
        )

        print_result(
            f"{metric.upper()} — INSTANCE LEVEL",
            result,
        )

    # =====================================================
    # Instance-level table
    # =====================================================

    print()
    print("=" * 60)
    print("INSTANCE-LEVEL MEANS")
    print("=" * 60)

    hv_baseline, hv_adaptive, _ = (
        extract_instance_level(
            data,
            "hypervolume",
        )
    )

    igd_baseline, igd_adaptive, _ = (
        extract_instance_level(
            data,
            "igd_plus",
        )
    )

    for i, instance in enumerate(
        INSTANCES
    ):

        print(
            f"{instance:15s} | "
            f"HV B={hv_baseline[i]:.6f} "
            f"CA={hv_adaptive[i]:.6f} | "
            f"IGD+ B={igd_baseline[i]:.6f} "
            f"CA={igd_adaptive[i]:.6f}"
        )

    print()
    print(
        "CROSS-INSTANCE ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()