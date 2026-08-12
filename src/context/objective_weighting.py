from __future__ import annotations

from dataclasses import dataclass

from src.context.context import Context


@dataclass(slots=True)
class ObjectiveWeights:
    makespan: float
    cost: float
    carbon: float
    energy: float


class ObjectiveWeighting:
    """
    Computes adaptive objective weights
    from the current project context.
    """

    def compute(
        self,
        context: Context,
    ) -> ObjectiveWeights:

        w_makespan = 1.0 + context.schedule_pressure

        w_cost = 1.0 + context.cost_pressure

        w_carbon = 1.0 + context.carbon_pressure

        w_energy = 1.0 + context.energy_pressure

        total = (
            w_makespan
            + w_cost
            + w_carbon
            + w_energy
        )

        return ObjectiveWeights(
            makespan=w_makespan / total,
            cost=w_cost / total,
            carbon=w_carbon / total,
            energy=w_energy / total,
        )