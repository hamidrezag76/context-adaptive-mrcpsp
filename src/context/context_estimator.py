"""
context_estimator.py

Computes project context dynamically
from the current project state.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from src.context.context import Context
from src.evaluation.result import EvaluationResult


class ContextEstimator:
    """
    Estimates project context using
    scheduling and sustainability indicators.
    """

    def __init__(self, project):

        self.project = project

    # --------------------------------------------------

    def estimate(

        self,

        evaluation: EvaluationResult,

    ) -> Context:

        makespan_ratio = (
            evaluation.makespan /
            self.project.horizon
        )

        cost_ratio = (
            evaluation.total_cost /
            self.project.reference_cost
        )

        carbon_ratio = (
            evaluation.total_carbon /
            self.project.reference_carbon
        )

        energy_ratio = (
            evaluation.total_energy /
            self.project.reference_energy
        )

        resource_pressure = min(
            1.0,
            makespan_ratio
        )

        uncertainty = (
            resource_pressure +
            carbon_ratio
        ) / 2.0

        return Context(

            carbon_pressure=min(1.0, carbon_ratio),

            energy_pressure=min(1.0, energy_ratio),

            resource_pressure=min(
                1.0,
                resource_pressure,
            ),

            cost_pressure=min(
                1.0,
                cost_ratio,
            ),

            schedule_pressure=min(
                1.0,
                makespan_ratio,
            ),

            uncertainty=min(
                1.0,
                uncertainty,
            ),
        )