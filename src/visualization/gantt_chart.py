"""
gantt_chart.py

Gantt Chart Visualization

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.scheduling.schedule import Schedule


class GanttChart:
    """
    Draw project Gantt Chart.
    """

    @staticmethod
    def plot(
        schedule: Schedule,
        output_file: Path,
    ) -> None:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig, ax = plt.subplots(figsize=(12, 8))

        activities = sorted(
            schedule.activities,
            key=lambda activity: activity.id,
        )

        for row, activity in enumerate(activities):

            ax.barh(
                y=row,
                width=activity.finish_time - activity.start_time,
                left=activity.start_time,
                height=0.6,
            )

        ax.set_yticks(
            range(len(activities))
        )

        ax.set_yticklabels(
            [
                f"A{activity.id}"
                for activity in activities
            ]
        )

        ax.set_xlabel("Time")

        ax.set_ylabel("Activities")

        ax.set_title("Project Gantt Chart")

        ax.grid(True)

        plt.tight_layout()

        plt.savefig(
            output_file,
            dpi=300,
        )

        plt.close()
