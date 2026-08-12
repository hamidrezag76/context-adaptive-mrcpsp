from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def main():

    print("\n========== EXPERIMENT METRICS PREPARATION ==========")

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    baseline = NSGA2(
        project,
        population_size=20,
        generations=20,
        seed=42,
        context_adaptive=False,
    )

    adaptive = NSGA2(
        project,
        population_size=20,
        generations=20,
        seed=42,
        context_adaptive=True,
    )

    baseline.run()
    adaptive.run()

    baseline_archive = baseline.archive.archive
    adaptive_archive = adaptive.archive.archive

    print(
        "Baseline archive:",
        len(baseline_archive),
    )

    print(
        "CA archive:",
        len(adaptive_archive),
    )

    baseline_points = [
        chromosome.objectives
        for chromosome in baseline_archive
    ]

    adaptive_points = [
        chromosome.objectives
        for chromosome in adaptive_archive
    ]

    print(
        "Baseline dimensions:",
        len(baseline_points[0]),
    )

    print(
        "CA dimensions:",
        len(adaptive_points[0]),
    )

    print(
        "Baseline first point:",
        baseline_points[0],
    )

    print(
        "CA first point:",
        adaptive_points[0],
    )

    assert baseline_points
    assert adaptive_points

    assert all(
        len(point) == 4
        for point in baseline_points
    )

    assert all(
        len(point) == 4
        for point in adaptive_points
    )

    print(
        "4D objective extraction: PASS"
    )


if __name__ == "__main__":
    main()