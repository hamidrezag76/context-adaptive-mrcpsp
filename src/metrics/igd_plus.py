"""
igd_plus.py

Inverted Generational Distance Plus (IGD+)
for multi-objective minimization problems.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence


ObjectiveVector = Sequence[float]


class IGDPlus:

    def __init__(
        self,
        reference_set: Iterable[ObjectiveVector],
    ) -> None:

        self.reference_set = [
            tuple(float(x) for x in point)
            for point in reference_set
        ]

        if not self.reference_set:
            raise ValueError(
                "Reference set cannot be empty."
            )

        dimensions = len(
            self.reference_set[0]
        )

        if dimensions == 0:
            raise ValueError(
                "Objective vectors cannot be empty."
            )

        for point in self.reference_set:

            if len(point) != dimensions:

                raise ValueError(
                    "Inconsistent objective dimensions "
                    "in reference set."
                )

        self.dimensions = dimensions

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def compute(
        self,
        approximation_set: Iterable[ObjectiveVector],
    ) -> float:

        approximation_set = [
            tuple(float(x) for x in point)
            for point in approximation_set
        ]

        if not approximation_set:

            raise ValueError(
                "Approximation set cannot be empty."
            )

        for point in approximation_set:

            if len(point) != self.dimensions:

                raise ValueError(
                    "Objective dimension mismatch."
                )

        total_distance = 0.0

        for reference_point in self.reference_set:

            minimum_distance = min(
                self._distance_plus(
                    reference_point,
                    approximation_point,
                )
                for approximation_point
                in approximation_set
            )

            total_distance += minimum_distance

        return (
            total_distance
            / len(self.reference_set)
        )

    # ---------------------------------------------------------
    # IGD+ distance
    # ---------------------------------------------------------

    @staticmethod
    def _distance_plus(
        reference_point: ObjectiveVector,
        approximation_point: ObjectiveVector,
    ) -> float:

        squared_distance = 0.0

        for reference, approximation in zip(
            reference_point,
            approximation_point,
        ):

            difference = (
                approximation - reference
            )

            squared_distance += max(
                0.0,
                difference,
            ) ** 2

        return sqrt(
            squared_distance
        )
