"""
reference_front.py

Reference Pareto Front Builder

Builds a global reference front from multiple optimization runs.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.optimization.chromosome import Chromosome


class ReferenceFrontBuilder:
    """
    Collects Pareto fronts and builds one global reference front.
    """

    def __init__(self) -> None:

        self._points: list[np.ndarray] = []

        self._all_points: list[np.ndarray] = []

    def add(
        self,
        pareto_front: Iterable[Chromosome],
    ) -> None:

        for chromosome in pareto_front:

            self._points.append(
                np.array(
                    [
                        chromosome.makespan,
                        chromosome.total_cost,
                        chromosome.total_carbon,
                        chromosome.total_energy,
                    ],
                    dtype=float,
                )
            )

    def add_population(
        self,
        population: Iterable[Chromosome],
    ) -> None:

        for chromosome in population:

            self._all_points.append(
                np.array(
                    [
                        chromosome.makespan,
                        chromosome.total_cost,
                        chromosome.total_carbon,
                        chromosome.total_energy,
                    ],
                    dtype=float,
                )
            )

    def build(
        self,
    ) -> np.ndarray:

        if not self._points:

            return np.empty(
                (
                    0,
                    4,
                ),
                dtype=float,
            )

        data = np.vstack(
            self._points,
        )

        mask = self._non_dominated_mask(
            data,
        )

        return data[
            mask
        ]

    @staticmethod
    def normalize(
        front: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        minimum = front.min(
            axis=0,
        )

        maximum = front.max(
            axis=0,
        )

        denominator = maximum - minimum

        denominator[
            denominator == 0
        ] = 1.0

        normalized = (
            front - minimum
        ) / denominator

        return (
            normalized,
            minimum,
            denominator,
        )

    @staticmethod
    def normalize_external(
        front: np.ndarray,
        minimum: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:

        normalized = (
            front - minimum
        ) / denominator
        
        manual = (front.max(axis=0) - minimum) / denominator

        return normalized

    def normalization_bounds(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:

        data = np.vstack(
            self._all_points
        )

        minimum = data.min(axis=0)

        maximum = data.max(axis=0)

        denominator = maximum - minimum

        denominator[denominator == 0] = 1.0

        return minimum, denominator

    @staticmethod
    def _non_dominated_mask(
        points: np.ndarray,
    ) -> np.ndarray:

        n = len(points)

        mask = np.ones(
            n,
            dtype=bool,
        )

        for i in range(n):

            if not mask[i]:
                continue

            for j in range(n):

                if i == j:
                    continue

                if (
                    np.all(points[j] <= points[i])
                    and np.any(points[j] < points[i])
                ):

                    mask[i] = False
                    break

        return mask
