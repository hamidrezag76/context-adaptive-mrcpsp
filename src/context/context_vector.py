"""
context_vector.py

Context Vector for the Context-Adaptive
Sustainable Multi-Mode RCPSP.

This class stores the current environmental
state of the optimization process.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ContextVector:
    """
    Represents the optimization context.

    All values are normalized into [0,1].
    """

    schedule_pressure: float

    resource_pressure: float

    cost_pressure: float

    carbon_pressure: float

    energy_pressure: float

    generation_progress: float

    population_diversity: float

    def as_tuple(
        self,
    ) -> tuple[float, ...]:
        """
        Return context as tuple.
        """

        return (
            self.schedule_pressure,
            self.resource_pressure,
            self.cost_pressure,
            self.carbon_pressure,
            self.energy_pressure,
            self.generation_progress,
            self.population_diversity,
        )
