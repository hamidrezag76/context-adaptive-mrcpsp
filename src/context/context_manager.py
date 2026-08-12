"""
context_manager.py

Population-driven Context Manager

Constructs the dynamic optimization context from
the evaluated population and project-level reference
values.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import statistics

from src.context.context import Context
from src.models.project import Project


class ContextManager:
    """
    Population-driven dynamic context manager.

    Context indicators are normalized to [0, 1].

    Project-level pressures are calculated against
    project reference values rather than against the
    instantaneous population range.
    """

    def __init__(
        self,
        project: Project | None = None,
        seed: int | None = None,
    ) -> None:

        self.project = project
        self.seed = seed

        self.current = Context.neutral()

    # =========================================================
    # Initialization
    # =========================================================

    def initialize(self) -> None:

        self.current = Context.neutral()

    # =========================================================
    # Update
    # =========================================================

    def update(
        self,
        population=None,
        generation: int = 0,
        max_generations: int = 1,
    ) -> Context:
        """
        Recalculate context from the evaluated population.
        """

        if population is None:

            return self.current

        chromosomes = list(
            getattr(
                population,
                "individuals",
                population,
            )
        )

        if not chromosomes:

            return self.current

        feasible = [
            chromosome
            for chromosome in chromosomes
            if getattr(
                chromosome,
                "feasible",
                True,
            )
        ]

        if not feasible:

            feasible = chromosomes

        # -----------------------------------------------------
        # Objective values
        # -----------------------------------------------------

        makespans = [
            float(c.makespan)
            for c in feasible
        ]

        costs = [
            float(c.total_cost)
            for c in feasible
        ]

        carbons = [
            float(c.total_carbon)
            for c in feasible
        ]

        energies = [
            float(c.total_energy)
            for c in feasible
        ]

        mean_makespan = statistics.mean(
            makespans
        )

        mean_cost = statistics.mean(
            costs
        )

        mean_carbon = statistics.mean(
            carbons
        )

        mean_energy = statistics.mean(
            energies
        )

        # -----------------------------------------------------
        # Project-relative pressures
        # -----------------------------------------------------

        schedule_pressure = self._ratio(
            mean_makespan,
            self._project_value(
                "horizon",
                1.0,
            ),
        )

        cost_pressure = self._ratio(
            mean_cost,
            self._project_value(
                "reference_cost",
                1.0,
            ),
        )

        carbon_pressure = self._ratio(
            mean_carbon,
            self._project_value(
                "reference_carbon",
                1.0,
            ),
        )

        energy_pressure = self._ratio(
            mean_energy,
            self._project_value(
                "reference_energy",
                1.0,
            ),
        )

        # -----------------------------------------------------
        # Resource pressure
        # -----------------------------------------------------

        resource_pressure = (
            self._resource_pressure(
                feasible
            )
        )

        # -----------------------------------------------------
        # Search uncertainty
        # -----------------------------------------------------

        uncertainty = (
            self._uncertainty(
                makespans,
                costs,
                carbons,
                energies,
            )
        )

        # -----------------------------------------------------
        # Context
        # -----------------------------------------------------

        self.current = Context(

            carbon_pressure=carbon_pressure,

            energy_pressure=energy_pressure,

            resource_pressure=resource_pressure,

            cost_pressure=cost_pressure,

            schedule_pressure=schedule_pressure,

            uncertainty=uncertainty,
        )

        self.current.clip()

        return self.current

    # =========================================================
    # Project value
    # =========================================================

    def _project_value(
        self,
        name: str,
        default: float,
    ) -> float:

        if self.project is None:

            return default

        value = getattr(
            self.project,
            name,
            default,
        )

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

        return max(
            value,
            1.0,
        )

    # =========================================================
    # Ratio
    # =========================================================

    @staticmethod
    def _ratio(
        value: float,
        reference: float,
    ) -> float:

        if reference <= 0:

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value / reference,
            ),
        )

    # =========================================================
    # Resource pressure
    # =========================================================

    @staticmethod
    def _resource_pressure(
        chromosomes,
    ) -> float:
        """
        Average peak renewable-resource utilization
        across the evaluated population.
        """

        pressures = []

        for chromosome in chromosomes:

            decoded = getattr(
                chromosome,
                "decoded_schedule",
                None,
            )

            if decoded is None:

                continue

            usage = getattr(
                decoded,
                "resource_usage",
                None,
            )

            capacities = getattr(
                decoded,
                "resource_capacities",
                None,
            )

            if not usage or not capacities:

                continue

            peak = 0.0

            for row in usage:

                for resource_id, capacity in enumerate(
                    capacities
                ):

                    if resource_id >= len(row):

                        continue

                    if capacity <= 0:

                        continue

                    utilization = (
                        float(row[resource_id])
                        / float(capacity)
                    )

                    peak = max(
                        peak,
                        utilization,
                    )

            pressures.append(
                max(
                    0.0,
                    min(
                        1.0,
                        peak,
                    ),
                )
            )

        if not pressures:

            return 0.0

        return float(
            statistics.mean(
                pressures
            )
        )

    # =========================================================
    # Uncertainty
    # =========================================================

    @staticmethod
    def _uncertainty(
        makespans,
        costs,
        carbons,
        energies,
    ) -> float:
        """
        Estimate search uncertainty from population
        dispersion.

        Coefficient of variation is calculated for
        each objective and then aggregated.
        """

        objective_groups = [
            makespans,
            costs,
            carbons,
            energies,
        ]

        coefficients = []

        for values in objective_groups:

            if len(values) <= 1:

                coefficients.append(
                    0.0
                )

                continue

            mean = statistics.mean(
                values
            )

            if mean <= 0:

                coefficients.append(
                    0.0
                )

                continue

            standard_deviation = (
                statistics.pstdev(
                    values
                )
            )

            coefficient = (
                standard_deviation
                / mean
            )

            coefficients.append(
                coefficient
            )

        if not coefficients:

            return 0.0

        # CV can theoretically exceed 1 for highly
        # dispersed populations. Clip the aggregate.
        return float(
            max(
                0.0,
                min(
                    1.0,
                    statistics.mean(
                        coefficients
                    ),
                ),
            )
        )

    # =========================================================
    # Access
    # =========================================================

    def get(self) -> Context:

        return self.current
