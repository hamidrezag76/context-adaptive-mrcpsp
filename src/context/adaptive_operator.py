"""
adaptive_operator.py

Context-Adaptive Operator Controller

This module dynamically adjusts crossover
and mutation probabilities according to the
current optimization context.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass

from src.context.context import Context


@dataclass(slots=True)
class OperatorSetting:

    crossover_probability: float

    mutation_probability: float


class AdaptiveOperator:

    """
    Adaptive operator controller.
    """

    def __init__(

        self,

        base_pc: float = 0.90,

        base_pm: float = 0.10,

    ):

        self.base_pc = base_pc

        self.base_pm = base_pm

    # -----------------------------------------------------

    def compute(

        self,

        context: Context,

    ) -> OperatorSetting:

        pc = self.base_pc

        pm = self.base_pm

        # ---------------------------------
        # Carbon pressure
        # ---------------------------------

        if context.carbon_pressure > 0.70:

            pm += 0.08

        # ---------------------------------
        # Energy pressure
        # ---------------------------------

        if context.energy_pressure > 0.70:

            pm += 0.05

        # ---------------------------------
        # Resource pressure
        # ---------------------------------

        if context.resource_pressure > 0.70:

            pm += 0.05

        # ---------------------------------
        # Cost pressure
        # ---------------------------------

        if context.cost_pressure > 0.70:

            pc -= 0.05

        # ---------------------------------
        # Schedule pressure
        # ---------------------------------

        if context.schedule_pressure > 0.70:

            pm += 0.05

        # ---------------------------------
        # Uncertainty
        # ---------------------------------

        if context.uncertainty > 0.70:

            pm += 0.10

            pc -= 0.05

        # ---------------------------------

        pc = max(

            0.50,

            min(

                0.95,

                pc,

            ),

        )

        pm = max(

            0.01,

            min(

                0.50,

                pm,

            ),

        )

        return OperatorSetting(

            crossover_probability=pc,

            mutation_probability=pm,

        )
