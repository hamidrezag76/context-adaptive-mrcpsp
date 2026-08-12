from __future__ import annotations

from src.experiments.statistical_analysis import (
    StatisticalAnalyzer,
)


def main() -> None:

    print()
    print(
        "========== STATISTICAL ANALYSIS TEST =========="
    )

    analyzer = StatisticalAnalyzer(
        alpha=0.05
    )

    # -------------------------------------------------
    # Synthetic paired data
    # -------------------------------------------------

    baseline_hv = [
        0.50,
        0.52,
        0.48,
        0.51,
        0.49,
        0.53,
        0.47,
        0.50,
        0.48,
        0.51,
    ]

    adaptive_hv = [
        0.60,
        0.62,
        0.58,
        0.61,
        0.59,
        0.63,
        0.57,
        0.60,
        0.58,
        0.61,
    ]

    hv = analyzer.compare(
        baseline_hv,
        adaptive_hv,
        "hypervolume",
    )

    print()
    print("HV test:")

    print(
        "  Baseline mean:",
        hv.baseline_mean,
    )

    print(
        "  Adaptive mean:",
        hv.adaptive_mean,
    )

    print(
        "  Relative improvement:",
        hv.relative_improvement_percent,
    )

    print(
        "  Wilcoxon p-value:",
        hv.wilcoxon_pvalue,
    )

    print(
        "  Paired t-test p-value:",
        hv.paired_t_pvalue,
    )

    print(
        "  Cohen's d:",
        hv.cohens_d,
    )

    print(
        "  Adaptive wins:",
        hv.better_adaptive_count,
    )

    print(
        "  Baseline wins:",
        hv.better_baseline_count,
    )

    assert (
        hv.adaptive_mean
        > hv.baseline_mean
    )

    assert (
        hv.relative_improvement_percent
        > 0.0
    )

    assert (
        hv.better_adaptive_count
        == 10
    )

    assert (
        hv.better_baseline_count
        == 0
    )

    assert (
        hv.wilcoxon_pvalue
        < 0.05
    )

    # -------------------------------------------------
    # IGD+ test
    # -------------------------------------------------

    baseline_igd = [
        0.20,
        0.22,
        0.19,
        0.21,
        0.23,
        0.20,
        0.24,
        0.21,
        0.22,
        0.20,
    ]

    adaptive_igd = [
        0.15,
        0.17,
        0.14,
        0.16,
        0.18,
        0.15,
        0.19,
        0.16,
        0.17,
        0.15,
    ]

    igd = analyzer.compare(
        baseline_igd,
        adaptive_igd,
        "igd_plus",
    )

    print()
    print("IGD+ test:")

    print(
        "  Baseline mean:",
        igd.baseline_mean,
    )

    print(
        "  Adaptive mean:",
        igd.adaptive_mean,
    )

    print(
        "  Relative improvement:",
        igd.relative_improvement_percent,
    )

    print(
        "  Wilcoxon p-value:",
        igd.wilcoxon_pvalue,
    )

    print(
        "  Paired t-test p-value:",
        igd.paired_t_pvalue,
    )

    print(
        "  Cohen's d:",
        igd.cohens_d,
    )

    assert (
        igd.adaptive_mean
        < igd.baseline_mean
    )

    assert (
        igd.relative_improvement_percent
        > 0.0
    )

    assert (
        igd.better_adaptive_count
        == 10
    )

    assert (
        igd.better_baseline_count
        == 0
    )

    assert (
        igd.wilcoxon_pvalue
        < 0.05
    )

    # -------------------------------------------------
    # Input validation
    # -------------------------------------------------

    try:

        analyzer.compare(
            [1.0, 2.0],
            [1.0],
            "hypervolume",
        )

    except ValueError:

        print(
            "Length validation: PASS"
        )

    else:

        raise AssertionError(
            "Length validation failed."
        )

    try:

        analyzer.compare(
            [1.0, 2.0],
            [1.0, 2.0],
            "unknown",
        )

    except ValueError:

        print(
            "Metric validation: PASS"
        )

    else:

        raise AssertionError(
            "Metric validation failed."
        )

    print()
    print(
        "STATISTICAL ANALYSIS TEST: PASS"
    )


if __name__ == "__main__":
    main()
