"""
Manual test for Performance Metrics.

Uses the current NSGA-II API and external EliteArchive.
"""

from pathlib import Path

from src.metrics.performance_metrics import PerformanceMetrics
from src.optimization.ca_nsga2 import NSGA2
from src.parser.psplib_parser import PSPLIBParser


def main() -> None:

    parser = PSPLIBParser()

    project = parser.load(
        Path(r"C:\Research\benchmarks\data\j301_1.mm")
    )

    optimizer = NSGA2(
        project=project,
        population_size=20,
        generations=5,
        seed=42,
        context_adaptive=False,
    )

    optimizer.run()

    pareto = optimizer.archive.archive

    print("=" * 60)
    print("Pareto Solutions")
    print("=" * 60)

    for chromosome in pareto:

        print(
            chromosome.makespan,
            chromosome.total_cost,
            chromosome.total_carbon,
            chromosome.total_energy,
        )

    print()

    reference = PerformanceMetrics.default_reference_point(
        pareto,
    )

    hypervolume = PerformanceMetrics.hypervolume(
        pareto,
        reference,
    )

    spacing = PerformanceMetrics.spacing(
        pareto,
    )

    print("=" * 60)
    print("Metrics")
    print("=" * 60)

    print("Reference Point :", reference)
    print("Hypervolume     :", hypervolume)
    print("Spacing         :", spacing)


if __name__ == "__main__":
    main()