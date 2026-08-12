from __future__ import annotations

from src.experiments.statistical_analysis import (
    StatisticalAnalyzer,
)


SEEDS = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
]


# ---------------------------------------------------------
# Real HV results from multi-seed experiment
# ---------------------------------------------------------

BASELINE_HV = [
    0.487514,
    0.889269,
    0.666718,
    0.741036,
    0.704717,
    0.282741,
    0.684228,
    0.702791,
    0.734972,
    0.653563,
]

CA_HV = [
    0.687318,
    0.810253,
    0.642918,
    0.645588,
    0.530458,
    0.638315,
    0.810570,
    0.690509,
    0.508941,
    0.717037,
]


# ---------------------------------------------------------
# Real IGD+ results from multi-seed experiment
# ---------------------------------------------------------

BASELINE_IGD = [
    0.237639,
    0.048291,
    0.138961,
    0.099995,
    0.139849,
    0.446224,
    0.112523,
    0.131485,
    0.101548,
    0.170830,
]

CA_IGD = [
    0.126681,
    0.062684,
    0.153976,
    0.149462,
    0.224871,
    0.138927,
    0.062378,
    0.130884,
    0.239761,
    0.115697,
]


def print_result(result) -> None:

    print()
    print(
        f"Metric: {result.metric}"
    )

    print(
        f"Baseline mean: "
        f"{result.baseline_mean:.8f}"
    )

    print(
        f"Adaptive mean: "
        f"{result.adaptive_mean:.8f}"
    )

    print(
        f"Baseline SD: "
        f"{result.baseline_std:.8f}"
    )

    print(
        f"Adaptive SD: "
        f"{result.adaptive_std:.8f}"
    )

    print(
        f"Mean difference: "
        f"{result.mean_difference:.8f}"
    )

    print(
        f"Median difference: "
        f"{result.median_difference:.8f}"
    )

    print(
        f"CA wins: "
        f"{result.better_adaptive_count}"
    )

    print(
        f"Baseline wins: "
        f"{result.better_baseline_count}"
    )

    print(
        f"Ties: "
        f"{result.ties}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{result.wilcoxon_statistic:.8f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{result.wilcoxon_pvalue:.8f}"
    )

    print(
        f"Paired t statistic: "
        f"{result.paired_t_statistic:.8f}"
    )

    print(
        f"Paired t p-value: "
        f"{result.paired_t_pvalue:.8f}"
    )

    print(
        f"Cohen's d: "
        f"{result.cohens_d:.8f}"
    )

    print(
        f"Relative improvement: "
        f"{result.relative_improvement_percent:.4f}%"
    )

    print(
        f"Significant at alpha=0.05: "
        f"{result.statistically_significant}"
    )


def main() -> None:

    print()
    print(
        "========== REAL STATISTICAL ANALYSIS =========="
    )

    print(
        "Seeds:",
        SEEDS,
    )

    assert len(BASELINE_HV) == len(SEEDS)
    assert len(CA_HV) == len(SEEDS)

    assert len(BASELINE_IGD) == len(SEEDS)
    assert len(CA_IGD) == len(SEEDS)

    analyzer = StatisticalAnalyzer(
        alpha=0.05
    )

    hv = analyzer.compare(
        BASELINE_HV,
        CA_HV,
        "hypervolume",
    )

    igd = analyzer.compare(
        BASELINE_IGD,
        CA_IGD,
        "igd_plus",
    )

    print_result(hv)
    print_result(igd)

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    assert (
        hv.baseline_mean
        > 0.0
    )

    assert (
        hv.adaptive_mean
        > 0.0
    )

    assert (
        igd.baseline_mean
        > 0.0
    )

    assert (
        igd.adaptive_mean
        > 0.0
    )

    assert (
        0.0 <= hv.wilcoxon_pvalue <= 1.0
    )

    assert (
        0.0 <= igd.wilcoxon_pvalue <= 1.0
    )

    assert (
        hv.better_adaptive_count
        + hv.better_baseline_count
        + hv.ties
        == len(SEEDS)
    )

    assert (
        igd.better_adaptive_count
        + igd.better_baseline_count
        + igd.ties
        == len(SEEDS)
    )

    print()
    print(
        "REAL STATISTICAL ANALYSIS: PASS"
    )


if __name__ == "__main__":
    main()