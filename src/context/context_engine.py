"""
context_engine.py

Context Extraction Engine

Computes dynamic project context from
current scheduling and sustainability state.

## Author

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from src.context.context import Context
from src.models.project import Project


class ContextEngine:
    """
    Extract dynamic optimization context.

    All returned indicators are normalized into [0,1].

    Resource pressure is calculated from the actual
    renewable-resource utilization of the decoded schedule.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:

        self.project = project

    # ------------------------------------------------------------
    # Main Context Evaluation
    # ------------------------------------------------------------

    def evaluate(
        self,
        makespan: float,
        total_cost: float,
        total_carbon: float,
        total_energy: float,
        resource_usage=None,
        resource_capacities=None,
    ) -> Context:
        """
        Compute the current optimization context.

        Resource pressure is derived from actual schedule
        utilization when resource usage data is available.
        """

        carbon_pressure = self._carbon_pressure(
            total_carbon,
        )

        energy_pressure = self._energy_pressure(
            total_energy,
        )

        resource_pressure = self._resource_pressure(
            resource_usage,
            resource_capacities,
        )

        cost_pressure = self._cost_pressure(
            total_cost,
        )

        schedule_pressure = self._schedule_pressure(
            makespan,
        )

        uncertainty = self._uncertainty()

        context = Context(
            carbon_pressure=carbon_pressure,
            energy_pressure=energy_pressure,
            resource_pressure=resource_pressure,
            cost_pressure=cost_pressure,
            schedule_pressure=schedule_pressure,
            uncertainty=uncertainty,
        )

        context.clip()

        return context

    # ------------------------------------------------------------
    # Carbon Pressure
    # ------------------------------------------------------------

    def _carbon_pressure(
        self,
        carbon: float,
    ) -> float:

        baseline = max(
            1.0,
            self.project.baseline_carbon,
        )

        return min(
            1.0,
            max(
                0.0,
                carbon / baseline,
            ),
        )

    # ------------------------------------------------------------
    # Energy Pressure
    # ------------------------------------------------------------

    def _energy_pressure(
        self,
        energy: float,
    ) -> float:

        baseline = max(
            1.0,
            self.project.baseline_energy,
        )

        return min(
            1.0,
            max(
                0.0,
                energy / baseline,
            ),
        )

    # ------------------------------------------------------------
    # Cost Pressure
    # ------------------------------------------------------------

    def _cost_pressure(
        self,
        cost: float,
    ) -> float:

        baseline = max(
            1.0,
            self.project.baseline_cost,
        )

        return min(
            1.0,
            max(
                0.0,
                cost / baseline,
            ),
        )

    # ------------------------------------------------------------
    # Schedule Pressure
    # ------------------------------------------------------------

    def _schedule_pressure(
        self,
        makespan: float,
    ) -> float:

        baseline = max(
            1.0,
            self.project.horizon,
        )

        return min(
            1.0,
            max(
                0.0,
                makespan / baseline,
            ),
        )

    # ------------------------------------------------------------
    # Resource Pressure
    # ------------------------------------------------------------

    def _resource_pressure(
        self,
        resource_usage,
        resource_capacities,
    ) -> float:
        """
        Calculate actual renewable-resource pressure.

        The pressure is defined as the maximum observed
        utilization ratio over all renewable resources
        and all active time periods:

            pressure =
                max_t,r(
                    usage[t,r] / capacity[r]
                )

        This captures the most constrained resource-time
        combination in the decoded schedule.

        If utilization data is unavailable, the method
        returns 0.0 rather than estimating pressure from
        static capacity alone.
        """

        if not resource_usage:
            return 0.0

        if not resource_capacities:
            return 0.0

        peak_utilization = 0.0

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

                peak_utilization = max(
                    peak_utilization,
                    utilization,
                )

        return min(
            1.0,
            max(
                0.0,
                peak_utilization,
            ),
        )

    # ------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------

    def _uncertainty(
        self,
    ) -> float:
        """
        Current uncertainty baseline.

        This remains a placeholder until the
        empirical uncertainty model is integrated.
        """

        return 0.50
