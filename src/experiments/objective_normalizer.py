"""
objective_normalizer.py

Common objective-space normalization for
comparative multi-objective experiments.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


ObjectiveVector = Sequence[float]


@dataclass(frozen=True, slots=True)
class NormalizationBounds:
    """
    Common lower and upper bounds for all objectives.
    """

    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    @property
    def dimensions(self) -> int:
        return len(self.minimum)


class ObjectiveNormalizer:
    """
    Normalize multi-objective minimization vectors
    using common bounds.

    The bounds must be constructed from the union
    of the compared solution sets.
    """

    def __init__(
        self,
        bounds: NormalizationBounds,
    ) -> None:

        if not bounds.minimum:
            raise ValueError(
                "Normalization bounds cannot be empty."
            )

        if len(bounds.minimum) != len(
            bounds.maximum
        ):
            raise ValueError(
                "Minimum and maximum dimensions "
                "must match."
            )

        for minimum, maximum in zip(
            bounds.minimum,
            bounds.maximum,
        ):

            if maximum < minimum:
                raise ValueError(
                    "Maximum bound cannot be smaller "
                    "than minimum bound."
                )

        self.bounds = bounds

    # ---------------------------------------------------------
    # Construct common bounds
    # ---------------------------------------------------------

    @classmethod
    def from_sets(
        cls,
        *solution_sets: Iterable[ObjectiveVector],
    ) -> "ObjectiveNormalizer":

        sets = [
            list(solution_set)
            for solution_set in solution_sets
        ]

        if not sets:
            raise ValueError(
                "At least one solution set is required."
            )

        if any(
            not solution_set
            for solution_set in sets
        ):
            raise ValueError(
                "Solution sets cannot be empty."
            )

        points = [
            tuple(float(x) for x in point)
            for solution_set in sets
            for point in solution_set
        ]

        dimensions = len(points[0])

        if dimensions == 0:
            raise ValueError(
                "Objective vectors cannot be empty."
            )

        if any(
            len(point) != dimensions
            for point in points
        ):
            raise ValueError(
                "Inconsistent objective dimensions."
            )

        minimum = tuple(
            min(
                point[d]
                for point in points
            )
            for d in range(dimensions)
        )

        maximum = tuple(
            max(
                point[d]
                for point in points
            )
            for d in range(dimensions)
        )

        return cls(
            NormalizationBounds(
                minimum=minimum,
                maximum=maximum,
            )
        )

    # ---------------------------------------------------------
    # Normalize one point
    # ---------------------------------------------------------

    def normalize(
        self,
        point: ObjectiveVector,
    ) -> tuple[float, ...]:

        point = tuple(
            float(x)
            for x in point
        )

        if len(point) != self.bounds.dimensions:
            raise ValueError(
                "Objective dimension mismatch."
            )

        result = []

        for value, minimum, maximum in zip(
            point,
            self.bounds.minimum,
            self.bounds.maximum,
        ):

            span = maximum - minimum

            if span == 0.0:
                result.append(0.0)

            else:
                result.append(
                    (value - minimum) / span
                )

        return tuple(result)

    # ---------------------------------------------------------
    # Normalize a set
    # ---------------------------------------------------------

    def normalize_set(
        self,
        points: Iterable[ObjectiveVector],
    ) -> list[tuple[float, ...]]:

        return [
            self.normalize(point)
            for point in points
        ]