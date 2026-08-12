"""
reference_set.py

Common reference-set construction for comparative
multi-objective experiments.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from typing import Iterable, Sequence


ObjectiveVector = Sequence[float]


class ReferenceSetBuilder:
    """
    Builds a common nondominated reference set from
    multiple approximation sets.
    """

    @staticmethod
    def dominates(
        a: ObjectiveVector,
        b: ObjectiveVector,
    ) -> bool:

        return (
            all(
                x <= y
                for x, y in zip(a, b)
            )
            and any(
                x < y
                for x, y in zip(a, b)
            )
        )

    @classmethod
    def nondominated(
        cls,
        points: Iterable[ObjectiveVector],
    ) -> list[tuple[float, ...]]:

        points = [
            tuple(float(x) for x in point)
            for point in points
        ]

        if not points:
            return []

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

        unique = list(
            dict.fromkeys(points)
        )

        result = []

        for point in unique:

            dominated = False

            for other in unique:

                if other == point:
                    continue

                if cls.dominates(
                    other,
                    point,
                ):
                    dominated = True
                    break

            if not dominated:
                result.append(point)

        return result

    @classmethod
    def build(
        cls,
        *approximation_sets: Iterable[ObjectiveVector],
    ) -> list[tuple[float, ...]]:

        points = [
            tuple(float(x) for x in point)
            for approximation_set in approximation_sets
            for point in approximation_set
        ]

        return cls.nondominated(points)

    @staticmethod
    def reference_point(
        reference_set: Iterable[ObjectiveVector],
        margin: float = 0.05,
    ) -> tuple[float, ...]:

        reference_set = [
            tuple(float(x) for x in point)
            for point in reference_set
        ]

        if not reference_set:
            raise ValueError(
                "Reference set cannot be empty."
            )

        if margin <= 0.0:
            raise ValueError(
                "Margin must be positive."
            )

        dimensions = len(reference_set[0])

        if any(
            len(point) != dimensions
            for point in reference_set
        ):
            raise ValueError(
                "Inconsistent objective dimensions."
            )

        maximum = [
            max(
                point[d]
                for point in reference_set
            )
            for d in range(dimensions)
        ]

        return tuple(
            value + margin
            for value in maximum
        )