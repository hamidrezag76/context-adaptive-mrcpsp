"""
operator_controller.py

Context-Adaptive Operator Controller

Dynamically adjusts crossover and mutation rates
according to the complete project context.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from src.context.context import Context


class OperatorController:
    """
    Context-adaptive controller for genetic operators.

    The controller maps the normalized project context
    to crossover and mutation probabilities.

    Higher environmental/economic pressure increases
    mutation to promote exploration.

    Higher uncertainty also increases mutation and
    reduces crossover slightly.

    All probabilities are explicitly bounded.
    """

    def __init__(self) -> None:

        self.base_crossover = 0.90

        self.base_mutation = 0.15

    # ---------------------------------------------------------
    # Crossover
    # ---------------------------------------------------------

    def crossover_probability(
        self,
        context: Context,
    ) -> float:
        """
        Compute adaptive crossover probability.

        Crossover is reduced under high uncertainty,
        resource pressure, cost pressure, and strong
        environmental pressure.
        """

        value = self.base_crossover

        # Uncertainty
        value -= 0.12 * context.uncertainty

        # Resource pressure
        value -= 0.08 * context.resource_pressure

        # Schedule pressure
        value -= 0.05 * context.schedule_pressure

        # Cost pressure
        value -= 0.04 * context.cost_pressure

        # Carbon pressure
        value -= 0.04 * context.carbon_pressure

        # Energy pressure
        value -= 0.03 * context.energy_pressure

        return max(
            0.50,
            min(
                0.95,
                value,
            ),
        )

    # ---------------------------------------------------------
    # Mutation
    # ---------------------------------------------------------

    def mutation_probability(
        self,
        context: Context,
    ) -> float:
        """
        Compute adaptive mutation probability.

        Mutation increases when the optimization
        environment becomes more demanding or uncertain.
        """

        value = self.base_mutation

        # Uncertainty
        value += 0.20 * context.uncertainty

        # Resource pressure
        value += 0.12 * context.resource_pressure

        # Schedule pressure
        value += 0.10 * context.schedule_pressure

        # Carbon pressure
        value += 0.08 * context.carbon_pressure

        # Energy pressure
        value += 0.06 * context.energy_pressure

        # Cost pressure
        value += 0.05 * context.cost_pressure

        return max(
            0.05,
            min(
                0.60,
                value,
            ),
        )
