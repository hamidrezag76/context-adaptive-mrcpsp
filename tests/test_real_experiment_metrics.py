from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2

from src.experiments.experiment_metrics import (
    ExperimentMetrics,
)


def main():

    print(
        "\n========== REAL EXPERIMENT METRICS =========="
    )

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    # -------------------------------------------------
    # Baseline
    # -------------------------------------------------

    baseline = NSGA2(
        project,
        population_size=20,
        generations=20,
        seed=42,
        context_adaptive=False,
    )

    baseline.run()

    # -------------------------------------------------
    # Context-Adaptive NSGA-II
    # -------------------------------------------------

    adaptive = NSGA2(
        project,
        population_size=20,
        generations=20,
        seed=42,
        context_adaptive=True,
    )

    adaptive.run()

    # -------------------------------------------------
    # Extract archives
    # -------------------------------------------------

    baseline_points = [
        chromosome.objectives
        for chromosome
        in baseline.archive.archive
    ]

    adaptive_points = [
        chromosome.objectives
        for chromosome
        in adaptive.archive.archive
    ]

    print(
        "Baseline archive size:",
        len(baseline_points),
    )

    print(
        "CA archive size:",
        len(adaptive_points),
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    evaluator = ExperimentMetrics(
        baseline_points,
        adaptive_points,
    )

    print(
        "Common reference set size:",
        len(evaluator.reference_set),
    )

    print(
        "Reference point:",
        evaluator.reference_point,
    )

    results = evaluator.evaluate()

    print(
        "\n========== METRICS =========="
    )

    print(
        "Baseline HV:",
        results["baseline_hv"],
    )

    print(
        "CA HV:",
        results["adaptive_hv"],
    )

    print(
        "Baseline IGD+:",
        results["baseline_igd_plus"],
    )

    print(
        "CA IGD+:",
        results["adaptive_igd_plus"],
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    assert len(evaluator.reference_set) > 0

    assert (
        results["baseline_hv"] >= 0.0
    )

    assert (
        results["adaptive_hv"] >= 0.0
    )

    assert (
        results["baseline_igd_plus"] >= 0.0
    )

    assert (
        results["adaptive_igd_plus"] >= 0.0
    )

    assert (
        len(evaluator.reference_point) == 4
    )

    print(
        "\nREAL EXPERIMENT METRICS: PASS"
    )


if __name__ == "__main__":
    main()