from __future__ import annotations

from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2

from src.experiments.metrics_evaluator import (
    MultiSeedMetricsEvaluator,
)


INSTANCE = Path(
    "benchmarks/data/j3010_1.mm"
)

SEEDS = list(
    range(42, 52)
)

POPULATION_SIZE = 20

GENERATIONS = 20


def run_case(
    project,
    context_adaptive: bool,
    seed: int,
) -> list[tuple[float, ...]]:

    optimizer = NSGA2(
        project,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seed=seed,
        context_adaptive=context_adaptive,
    )

    optimizer.run()

    return [
        chromosome.objectives
        for chromosome
        in optimizer.archive.archive
    ]


def main() -> None:

    print()
    print(
        "========== MULTI-SEED METRICS =========="
    )

    print(
        "Instance:",
        INSTANCE.name,
    )

    print(
        "Population:",
        POPULATION_SIZE,
    )

    print(
        "Generations:",
        GENERATIONS,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print()

    project = MMParser(
        INSTANCE
    ).parse()

    baseline_archives = {}

    adaptive_archives = {}

    # -------------------------------------------------
    # Run all seeds
    # -------------------------------------------------

    for seed in SEEDS:

        print(
            f"Running seed {seed} ..."
        )

        baseline_archives[seed] = run_case(
            project,
            context_adaptive=False,
            seed=seed,
        )

        adaptive_archives[seed] = run_case(
            project,
            context_adaptive=True,
            seed=seed,
        )

        print(
            f"  Baseline archive: "
            f"{len(baseline_archives[seed])}"
        )

        print(
            f"  CA archive:       "
            f"{len(adaptive_archives[seed])}"
        )

    # -------------------------------------------------
    # Common multi-seed metrics
    # -------------------------------------------------

    evaluator = MultiSeedMetricsEvaluator(
        baseline_archives=baseline_archives,
        adaptive_archives=adaptive_archives,
    )

    print()
    print(
        "========== COMMON METRIC SETUP =========="
    )

    print(
        "Seeds evaluated:",
        evaluator.seeds,
    )

    print(
        "Common reference set size:",
        len(
            evaluator.reference_set
        ),
    )

    print(
        "Reference point:",
        evaluator.reference_point,
    )

    # -------------------------------------------------
    # Per-seed metrics
    # -------------------------------------------------

    results = evaluator.evaluate()

    print()
    print(
        "========== PER-SEED METRICS =========="
    )

    for result in results:

        seed = int(
            result["seed"]
        )

        print(
            f"Seed {seed:2d} | "
            f"Baseline HV="
            f"{result['baseline_hv']:.6f} | "
            f"CA HV="
            f"{result['adaptive_hv']:.6f} | "
            f"Baseline IGD+="
            f"{result['baseline_igd_plus']:.6f} | "
            f"CA IGD+="
            f"{result['adaptive_igd_plus']:.6f}"
        )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    summary = evaluator.summary()

    hv = summary["hypervolume"]

    igd = summary["igd_plus"]

    print()
    print(
        "========== SUMMARY =========="
    )

    print()
    print("Hypervolume")

    print(
        "  Baseline mean:",
        hv["baseline_mean"],
    )

    print(
        "  Baseline SD:",
        hv["baseline_std"],
    )

    print(
        "  CA mean:",
        hv["adaptive_mean"],
    )

    print(
        "  CA SD:",
        hv["adaptive_std"],
    )

    print(
        "  Relative improvement:",
        f"{hv['relative_improvement_percent']:.2f}%"
    )

    print()
    print("IGD+")

    print(
        "  Baseline mean:",
        igd["baseline_mean"],
    )

    print(
        "  Baseline SD:",
        igd["baseline_std"],
    )

    print(
        "  CA mean:",
        igd["adaptive_mean"],
    )

    print(
        "  CA SD:",
        igd["adaptive_std"],
    )

    print(
        "  Relative improvement:",
        f"{igd['relative_improvement_percent']:.2f}%"
    )

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    assert len(results) == len(SEEDS)

    assert (
        len(evaluator.reference_set)
        > 0
    )

    assert (
        len(evaluator.reference_point)
        == 4
    )

    for result in results:

        assert (
            result["baseline_hv"]
            >= 0.0
        )

        assert (
            result["adaptive_hv"]
            >= 0.0
        )

        assert (
            result["baseline_igd_plus"]
            >= 0.0
        )

        assert (
            result["adaptive_igd_plus"]
            >= 0.0
        )

    print()
    print(
        "MULTI-SEED METRICS TEST: PASS"
    )


if __name__ == "__main__":
    main()
