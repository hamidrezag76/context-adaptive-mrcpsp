"""
main.py

CA-SMRCPSP Main Execution Pipeline

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

from src.evaluation.performance_metrics import PerformanceMetrics
from src.optimization.nsga2 import NSGAII
from src.parser.psplib_parser import PSPLIBParser
from src.visualization.convergence_plot import ConvergencePlot
from src.visualization.gantt_chart import GanttChart
from src.visualization.hypervolume_plot import HypervolumePlot
from src.visualization.igd_plot import IGDPlot
from src.visualization.pareto_plot import ParetoPlot


def run_project(
    benchmark: Path,
    output_directory: Path,
    population_size: int = 100,
    generations: int = 80,
) -> None:
    """
    Execute one benchmark.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    parser = PSPLIBParser()

    project = parser.load(
        benchmark,
    )

    optimizer = NSGAII(
        project=project,
        population_size=population_size,
        generations=generations,
    )

    pareto = optimizer.run()

    print(f"Pareto Solutions : {len(pareto)}")

    # -------------------------
    # Performance Metrics
    # -------------------------

    metrics = PerformanceMetrics()

    print(metrics.summary(pareto))

    # -------------------------
    # Pareto Plot
    # -------------------------

    ParetoPlot.plot(
        pareto,
        output_directory / "pareto_front.png",
    )

    # -------------------------
    # Convergence
    # -------------------------

    if hasattr(optimizer, "best_history"):

        ConvergencePlot.plot(
            optimizer.best_history,
            ylabel="Best Makespan",
            title="Convergence",
            output_file=output_directory / "convergence.png",
        )

    # -------------------------
    # Hypervolume
    # -------------------------

    if hasattr(optimizer, "hypervolume_history"):

        HypervolumePlot.plot(
            optimizer.hypervolume_history,
            output_directory / "hypervolume.png",
        )

    # -------------------------
    # IGD
    # -------------------------

    if hasattr(optimizer, "igd_history"):

        IGDPlot.plot(
            optimizer.igd_history,
            output_directory / "igd.png",
        )

    print("Execution Finished.")


if __name__ == "__main__":

    run_project(
        benchmark=Path(r"C:\Research\benchmarks\data\j301_1.mm"),
        output_directory=Path(r"C:\Research\outputs"),
    )
