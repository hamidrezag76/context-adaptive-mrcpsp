"""
hypervolume.py

Pareto-front Hypervolume indicator
for the CA-SMRCPSP experimental framework.

All objectives are minimization objectives.
"""

from __future__ import annotations

from typing import Iterable, Sequence


ObjectiveVector = Sequence[float]


class Hypervolume:

    def __init__(
        self,
        reference_point: ObjectiveVector,
    ) -> None:

        if len(reference_point) == 0:
            raise ValueError(
                "Reference point cannot be empty."
            )

        self.reference_point = tuple(
            float(x)
            for x in reference_point
        )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def compute(
        self,
        points: Iterable[ObjectiveVector],
    ) -> float:

        points = [
            tuple(float(x) for x in point)
            for point in points
        ]

        if not points:
            return 0.0

        dimensions = len(
            self.reference_point
        )

        for point in points:

            if len(point) != dimensions:
                raise ValueError(
                    "Objective dimension mismatch."
                )

            if any(
                x > r
                for x, r in zip(
                    point,
                    self.reference_point,
                )
            ):
                raise ValueError(
                    "A Pareto point is worse than "
                    "the reference point."
                )

        # Remove dominated points before calculation.
        points = self._filter_nondominated(
            points
        )

        if dimensions == 1:

            return max(
                0.0,
                self.reference_point[0]
                - min(point[0] for point in points),
            )

        if dimensions == 2:

            return self._compute_2d(
                points
            )

        return self._compute_recursive(
            points,
            self.reference_point,
        )

    # ---------------------------------------------------------
    # Non-dominated filtering
    # ---------------------------------------------------------

    @staticmethod
    def _dominates(
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
    def _filter_nondominated(
        cls,
        points: list[tuple[float, ...]],
    ) -> list[tuple[float, ...]]:

        result = []

        for i, point in enumerate(points):

            dominated = False

            for j, other in enumerate(points):

                if i == j:
                    continue

                if cls._dominates(
                    other,
                    point,
                ):
                    dominated = True
                    break

            if not dominated:
                result.append(point)

        return list(
            dict.fromkeys(result)
        )

    # ---------------------------------------------------------
    # 2D Hypervolume
    # ---------------------------------------------------------

    def _compute_2d(
        self,
        points,
    ) -> float:

        points = sorted(
            points,
            key=lambda p: p[0],
        )

        hv = 0.0

        previous_y = self.reference_point[1]

        for x, y in points:

            width = (
                self.reference_point[0]
                - x
            )

            height = (
                previous_y - y
            )

            if width > 0 and height > 0:

                hv += width * height

            previous_y = min(
                previous_y,
                y,
            )

        return hv

    # ---------------------------------------------------------
    # Recursive Hypervolume
    # ---------------------------------------------------------

    def _compute_recursive(
        self,
        points,
        reference,
    ) -> float:

        dimensions = len(reference)

        if dimensions == 1:

            return max(
                0.0,
                reference[0]
                - min(
                    point[0]
                    for point in points
                ),
            )

        points = sorted(
            points,
            key=lambda p: p[-1],
        )

        volume = 0.0

        previous = reference[-1]

        while points:

            current = points[-1]

            height = (
                previous
                - current[-1]
            )

            if height > 0:

                projected = [
                    point[:-1]
                    for point in points
                    if point[-1] <= current[-1]
                ]

                volume += (
                    self._compute_recursive(
                        projected,
                        reference[:-1],
                    )
                    * height
                )

            previous = current[-1]

            points = [
                point
                for point in points
                if point[-1] < previous
            ]

        return volume
