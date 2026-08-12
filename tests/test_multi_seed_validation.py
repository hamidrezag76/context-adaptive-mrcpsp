from __future__ import annotations

from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


INSTANCE = Path("benchmarks/data/j3010_1.mm")

SEEDS = list(range(42, 52))

POPULATION_SIZE = 20
GENERATIONS = 20


def run_case(
    context_adaptive: bool,
    seed: int,
):
    project = MMParser(INSTANCE).parse()

    optimizer = NSGA2(
        project,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seed=seed,
        context_adaptive=context_adaptive,
    )

    optimizer.run()

    individuals = optimizer.population.individuals

    best_makespan = min(
        c.makespan
        for c in individuals
    )

    best_cost = min(
        c.total_cost
        for c in individuals
    )

    best_carbon = min(
        c.total_carbon
        for c in individuals
    )

    best_energy = min(
        c.total_energy
        for c in individuals
    )

    return {
        "seed": seed,
        "best_makespan": best_makespan,
        "best_cost": best_cost,
        "best_carbon": best_carbon,
        "best_energy": best_energy,
        "archive_size": len(optimizer.archive.archive),
    }


def main():

    print()
    print("========== MULTI-SEED VALIDATION ==========")
    print(
        f"Instance: {INSTANCE.name}"
    )
    print(
        f"Population: {POPULATION_SIZE}"
    )
    print(
        f"Generations: {GENERATIONS}"
    )
    print(
        f"Seeds: {SEEDS}"
    )
    print()

    baseline_results = []
    adaptive_results = []

    for seed in SEEDS:

        baseline = run_case(
            context_adaptive=False,
            seed=seed,
        )

        adaptive = run_case(
            context_adaptive=True,
            seed=seed,
        )

        baseline_results.append(
            baseline
        )

        adaptive_results.append(
            adaptive
        )

        print(
            f"Seed {seed:2d} | "
            f"Baseline "
            f"M={baseline['best_makespan']:5.1f} "
            f"C={baseline['best_cost']:12.2f} "
            f"Carbon={baseline['best_carbon']:12.2f} "
            f"Energy={baseline['best_energy']:12.2f} "
            f"Archive={baseline['archive_size']:2d} | "
            f"CA "
            f"M={adaptive['best_makespan']:5.1f} "
            f"C={adaptive['best_cost']:12.2f} "
            f"Carbon={adaptive['best_carbon']:12.2f} "
            f"Energy={adaptive['best_energy']:12.2f} "
            f"Archive={adaptive['archive_size']:2d}"
        )

    print()
    print("========== SUMMARY ==========")

    metrics = [
        "best_makespan",
        "best_cost",
        "best_carbon",
        "best_energy",
        "archive_size",
    ]

    for metric in metrics:

        baseline_values = [
            r[metric]
            for r in baseline_results
        ]

        adaptive_values = [
            r[metric]
            for r in adaptive_results
        ]

        baseline_mean = (
            sum(baseline_values)
            / len(baseline_values)
        )

        adaptive_mean = (
            sum(adaptive_values)
            / len(adaptive_values)
        )

        print()
        print(metric)

        print(
            "  Baseline mean:",
            baseline_mean,
        )

        print(
            "  CA mean:",
            adaptive_mean,
        )

        if baseline_mean != 0:

            improvement = (
                (baseline_mean - adaptive_mean)
                / abs(baseline_mean)
            ) * 100.0

            print(
                "  Relative improvement:",
                f"{improvement:.2f}%"
            )


if __name__ == "__main__":
    main()
