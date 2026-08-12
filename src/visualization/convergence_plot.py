"""
convergence_plot.py

Convergence Curve Visualization

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class ConvergencePlot:
    """
    Plot best objective value through generations.
    """

    @staticmethod
    def plot(
        history: list[float],
        ylabel: str,
        title: str,
        output_file: Path,
    ) -> None:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            range(1, len(history) + 1),
            history,
            linewidth=2,
        )

        plt.xlabel("Generation")

        plt.ylabel(ylabel)

        plt.title(title)

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
        )

        plt.close()
