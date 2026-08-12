"""
chromosome.py

Chromosome representation for CA-SMRCPSP.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any

@dataclass(slots=True)
class Chromosome:
    """
    Chromosome representation.

    A chromosome consists of

        1. Activity priority list

        2. Mode assignment

    """

    priority_list: list[int]

    mode_assignment: dict[int, int]

    # -----------------------------
    # Objective values
    # -----------------------------

    makespan: float = 0.0

    total_cost: float = 0.0

    total_carbon: float = 0.0

    total_energy: float = 0.0

    # -----------------------------
    # NSGA-II Attributes
    # -----------------------------

    rank: int = 0

    crowding_distance: float = 0.0

    # -----------------------------
    # Constraint Handling
    # -----------------------------

    feasible: bool = True

    penalty: float = 0.0

    decoded_schedule: Any | None = None

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def copy(
        self,
    ) -> "Chromosome":
        """
        Deep copy chromosome.
        """

        return Chromosome(
            priority_list=self.priority_list.copy(),
            mode_assignment=self.mode_assignment.copy(),
            makespan=self.makespan,
            total_cost=self.total_cost,
            total_carbon=self.total_carbon,
            total_energy=self.total_energy,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            feasible=self.feasible,
            penalty=self.penalty,
            decoded_schedule=None,
        )

    @property
    def objectives(
        self,
    ) -> tuple[float, float, float, float]:
        """
        Multi-objective vector.
        """

        return (
            self.makespan,
            self.total_cost,
            self.total_carbon,
            self.total_energy,
        )

    def reset_objectives(
        self,
    ) -> None:
        """
        Reset objective values.
        """

        self.makespan = 0.0

        self.total_cost = 0.0

        self.total_carbon = 0.0

        self.total_energy = 0.0

        self.rank = 0

        self.crowding_distance = 0.0

        self.feasible = True

        self.penalty = 0.0

        self.decoded_schedule = None

    # NSGA-II attributes

    domination_count: int = 0

    dominated_solutions: list[int] = field(
        default_factory=list,
    )
