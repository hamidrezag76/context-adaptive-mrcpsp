"""
pareto_plot.py

Pareto Front Visualization

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.optimization.chromosome import Chromosome


class ParetoPlot:
    """
    Plot Pareto Fronts.
    """

    @staticmethod
    def plot(
        pareto_front: list[Chromosome],
        output_file: Path,
    ) -> None:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        makespan = [c.makespan for c in pareto_front]
        cost = [c.total_cost for c in pareto_front]

        plt.figure(figsize=(8, 6))

        plt.scatter(
            makespan,
            cost,
            s=40,
        )

        plt.xlabel("Makespan")

        plt.ylabel("Total Cost")

        plt.title("Pareto Front")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
        )

        plt.close()
