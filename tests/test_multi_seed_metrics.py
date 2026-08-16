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
    
def test_all_three_modes_share_common_metric_reference():

    from src.experiments.metrics_evaluator import (
        MultiSeedMetricsEvaluator,
    )

    baseline = {
        42: [
            (1.0, 4.0, 7.0, 10.0),
            (2.0, 3.0, 6.0, 9.0),
        ],
        43: [
            (1.5, 4.5, 7.5, 10.5),
            (2.5, 3.5, 6.5, 9.5),
        ],
    }

    context_only = {
        42: [
            (1.2, 3.8, 6.8, 9.8),
            (2.2, 2.8, 5.8, 8.8),
        ],
        43: [
            (1.7, 4.3, 7.3, 10.3),
            (2.7, 3.3, 6.3, 9.3),
        ],
    }

    adaptive = {
        42: [
            (0.8, 3.5, 6.5, 8.5),
            (1.8, 2.5, 5.5, 7.5),
        ],
        43: [
            (1.3, 4.0, 7.0, 9.0),
            (2.3, 3.0, 6.0, 8.0),
        ],
    }

    evaluator = MultiSeedMetricsEvaluator(
        baseline_archives=baseline,
        context_only_archives=context_only,
        adaptive_archives=adaptive,
    )

    assert evaluator.seeds == [42, 43]

    assert evaluator.reference_set

    assert len(
        evaluator.reference_point
    ) == 4

    results = evaluator.evaluate()

    assert len(results) == 2

    assert {
        int(result["seed"])
        for result in results
    } == {42, 43}

    for result in results:

        assert "baseline_hv" in result
        assert "context_only_hv" in result
        assert "adaptive_hv" in result

        assert "baseline_igd_plus" in result
        assert "context_only_igd_plus" in result
        assert "adaptive_igd_plus" in result

        for key, value in result.items():

            if key == "seed":
                continue

            import math

            assert math.isfinite(
                float(value)
            )
def test_common_reference_depends_on_all_three_modes():

    from src.experiments.metrics_evaluator import (
        MultiSeedMetricsEvaluator,
    )

    baseline = {
        42: [
            (1.0, 1.0, 1.0, 1.0),
        ],
    }

    context_only = {
        42: [
            (2.0, 2.0, 2.0, 2.0),
        ],
    }

    adaptive = {
        42: [
            (0.5, 0.5, 0.5, 0.5),
        ],
    }

    evaluator = MultiSeedMetricsEvaluator(
        baseline_archives=baseline,
        context_only_archives=context_only,
        adaptive_archives=adaptive,
    )

    # The adaptive point dominates both other points,
    # therefore the common nondominated reference set
    # must contain the adaptive point after normalization.
    assert len(
        evaluator.reference_set
    ) == 1

    # The common reference point must therefore be
    # constructed from the complete normalized union.
    assert len(
        evaluator.reference_point
    ) == 4

    # The adaptive point is strictly better in every
    # objective than baseline and context-only.
    adaptive_normalized = (
        evaluator.adaptive_normalized[42]
    )

    assert len(
        adaptive_normalized
    ) == 1

    assert tuple(
        evaluator.reference_set[0]
    ) == tuple(
        adaptive_normalized[0]
    )