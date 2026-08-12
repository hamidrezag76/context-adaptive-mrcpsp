from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2
from src.experiments.result_store import ResultStore


def main():

    print(
        "\n========== REAL RESULT STORE =========="
    )

    instance = Path(
        "benchmarks/data/j3010_1.mm"
    )

    population_size = 20
    generations = 20
    seed = 42

    project = MMParser(
        instance
    ).parse()

    # -------------------------------------------------
    # Baseline
    # -------------------------------------------------

    baseline = NSGA2(
        project,
        population_size=population_size,
        generations=generations,
        seed=seed,
        context_adaptive=False,
    )

    baseline.run()

    baseline_points = [
        chromosome.objectives
        for chromosome
        in baseline.archive.archive
    ]

    baseline_best = min(
        baseline.population.individuals,
        key=lambda chromosome:
            chromosome.makespan,
    )

    # -------------------------------------------------
    # CA-NSGA-II
    # -------------------------------------------------

    adaptive = NSGA2(
        project,
        population_size=population_size,
        generations=generations,
        seed=seed,
        context_adaptive=True,
    )

    adaptive.run()

    adaptive_points = [
        chromosome.objectives
        for chromosome
        in adaptive.archive.archive
    ]

    adaptive_best = min(
        adaptive.population.individuals,
        key=lambda chromosome:
            chromosome.makespan,
    )

    # -------------------------------------------------
    # Storage
    # -------------------------------------------------

    store = ResultStore(
        "results/raw"
    )

    baseline_path = store.save_run(
        instance=instance.name,
        algorithm="baseline_nsga2",
        seed=seed,
        population_size=population_size,
        generations=generations,
        archive_points=baseline_points,
        metrics={},
        best_objectives=baseline_best.objectives,
        metadata={
            "context_adaptive": False,
        },
        overwrite=True,
    )

    adaptive_path = store.save_run(
        instance=instance.name,
        algorithm="ca_nsga2",
        seed=seed,
        population_size=population_size,
        generations=generations,
        archive_points=adaptive_points,
        metrics={},
        best_objectives=adaptive_best.objectives,
        metadata={
            "context_adaptive": True,
        },
        overwrite=True,
    )

    print(
        "Baseline saved:",
        baseline_path,
    )

    print(
        "CA saved:",
        adaptive_path,
    )

    print(
        "Baseline archive:",
        len(baseline_points),
    )

    print(
        "CA archive:",
        len(adaptive_points),
    )

    # -------------------------------------------------
    # Reload
    # -------------------------------------------------

    records = store.find(
        instance=instance.name,
        seeds=[seed],
    )

    print(
        "Stored records:",
        len(records),
    )

    assert len(records) == 2

    print(
        "\nREAL RESULT STORAGE: PASS"
    )


if __name__ == "__main__":
    main()
