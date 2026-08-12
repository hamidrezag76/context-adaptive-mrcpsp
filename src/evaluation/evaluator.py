"""
evaluator.py

Multi-objective evaluation engine.

Evaluates one decoded schedule.
"""

from __future__ import annotations

from src.evaluation.result import EvaluationResult
from src.models.project import Project
from src.context.context import Context
from src.context.context_engine import ContextEngine


class Evaluator:
    """
    Multi-objective evaluator.

    Computes the four optimization objectives and
    extracts schedule-level resource pressure.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:

        self.project = project

        self.context_engine = ContextEngine(
            project,
        )

    # ---------------------------------------------------------
    # Main Evaluation
    # ---------------------------------------------------------

    def evaluate(
        self,
        schedule,
        context: Context | None = None,
    ) -> EvaluationResult:
        """
        Evaluate one decoded schedule.
        """

        makespan = self._calculate_makespan(
            schedule,
        )

        total_cost = self._calculate_total_cost(
            schedule,
        )

        total_carbon = self._calculate_total_carbon(
            schedule,
        )

        total_energy = self._calculate_total_energy(
            schedule,
        )

        resource_pressure = (
            self.context_engine._resource_pressure(
                schedule.resource_usage,
                schedule.resource_capacities,
            )
        )

        average_resource_utilization = (
            self._average_resource_utilization(
                schedule.resource_usage,
                schedule.resource_capacities,
            )
        )

        peak_resource_utilization = (
            resource_pressure
        )

        return EvaluationResult(
            makespan=makespan,
            total_cost=total_cost,
            total_carbon=total_carbon,
            total_energy=total_energy,
            feasible=schedule.feasible,
            resource_pressure=resource_pressure,
            average_resource_utilization=(
                average_resource_utilization
            ),
            peak_resource_utilization=(
                peak_resource_utilization
            ),
        )

    # ---------------------------------------------------------
    # Makespan
    # ---------------------------------------------------------

    def _calculate_makespan(
        self,
        decoded,
    ) -> float:

        return float(
            decoded.makespan
        )

    # ---------------------------------------------------------
    # Cost
    # ---------------------------------------------------------

    def _calculate_total_cost(
        self,
        decoded,
    ) -> float:

        total = 0.0

        for activity in (
            self.project.activities.values()
        ):

            mode = activity.get_mode(
                decoded.mode_assignment[
                    activity.id
                ]
            )

            total += mode.cost

        return total

    # ---------------------------------------------------------
    # Carbon
    # ---------------------------------------------------------

    def _calculate_total_carbon(
        self,
        decoded,
    ) -> float:

        total = 0.0

        for activity in (
            self.project.activities.values()
        ):

            mode = activity.get_mode(
                decoded.mode_assignment[
                    activity.id
                ]
            )

            total += mode.carbon

        return total

    # ---------------------------------------------------------
    # Energy
    # ---------------------------------------------------------

    def _calculate_total_energy(
        self,
        decoded,
    ) -> float:

        total = 0.0

        for activity in (
            self.project.activities.values()
        ):

            mode = activity.get_mode(
                decoded.mode_assignment[
                    activity.id
                ]
            )

            total += mode.energy

        return total

    # ---------------------------------------------------------
    # Average Resource Utilization
    # ---------------------------------------------------------

    @staticmethod
    def _average_resource_utilization(
        resource_usage,
        resource_capacities,
    ) -> float:
        """
        Calculate the mean utilization over all
        resource-time combinations.

        Only positive-capacity resources are included.
        """

        if not resource_usage:
            return 0.0

        if not resource_capacities:
            return 0.0

        values = []

        for time_row in resource_usage:

            for resource_index, usage in enumerate(
                time_row
            ):

                if resource_index >= len(
                    resource_capacities
                ):
                    continue

                capacity = float(
                    resource_capacities[
                        resource_index
                    ]
                )

                if capacity <= 0.0:
                    continue

                utilization = (
                    float(usage)
                    / capacity
                )

                values.append(
                    min(
                        1.0,
                        max(
                            0.0,
                            utilization,
                        ),
                    )
                )

        if not values:
            return 0.0

        return float(
            sum(values)
            / len(values)
        )
