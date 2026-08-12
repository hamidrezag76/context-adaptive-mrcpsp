from __future__ import annotations

from src.models.project import Project


class ProjectStatistics:

    def __init__(
        self,
        project: Project,
    ):
        self.project = project

    def compute(self):

        baseline_cost = 0.0
        baseline_carbon = 0.0
        baseline_energy = 0.0

        reference_cost = 0.0
        reference_carbon = 0.0
        reference_energy = 0.0

        for activity in self.project.activities.values():

            # ---------- Baseline ----------
            first_mode = activity.modes[0]

            baseline_cost += first_mode.cost
            baseline_carbon += first_mode.carbon
            baseline_energy += first_mode.energy

            # ---------- Reference ----------
            reference_cost += max(
                m.cost for m in activity.modes
            )

            reference_carbon += max(
                m.carbon for m in activity.modes
            )

            reference_energy += max(
                m.energy for m in activity.modes
            )

        self.project.baseline_cost = baseline_cost
        self.project.baseline_carbon = baseline_carbon
        self.project.baseline_energy = baseline_energy

        self.project.reference_cost = reference_cost
        self.project.reference_carbon = reference_carbon
        self.project.reference_energy = reference_energy