"""
igd_plot.py

Inverted Generational Distance (IGD) Evolution Plot

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class IGDPlot:
    """
    Plot IGD evolution.
    """

    @staticmethod
    def plot(
        igd_history: list[float],
        output_file: Path,
    ) -> None:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            range(1, len(igd_history) + 1),
            igd_history,
            linewidth=2,
        )

        plt.xlabel("Generation")

        plt.ylabel("IGD")

        plt.title("IGD Evolution")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
        )

        plt.close()
