"""
hypervolume_plot.py

Hypervolume Evolution Plot

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class HypervolumePlot:
    """
    Plot Hypervolume evolution.
    """

    @staticmethod
    def plot(
        hypervolume_history: list[float],
        output_file: Path,
    ) -> None:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            range(1, len(hypervolume_history) + 1),
            hypervolume_history,
            linewidth=2,
        )

        plt.xlabel("Generation")

        plt.ylabel("Hypervolume")

        plt.title("Hypervolume Evolution")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
        )

        plt.close()
