"""
performance_metrics.py

Performance metrics for multi-objective optimization.

Includes:

- Hypervolume (HV)
- Generational Distance (GD)
- Inverted Generational Distance (IGD)
- Spread (Delta)
- Spacing

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from math import sqrt

from src.optimization.chromosome import Chromosome


class PerformanceMetrics:
    """
    Quality indicators for Pareto fronts.
    """

    @staticmethod
    def hypervolume(
        pareto_front: list[Chromosome],
        reference_point: tuple[float, float],
    ) -> float:
        """
        Compute 2-objective Hypervolume.

        Assumes minimization.

        Objectives:
            Makespan
            Total Cost
        """

        if not pareto_front:
            return 0.0

        front = sorted(
            pareto_front,
            key=lambda c: c.makespan,
        )

        hv = 0.0

        previous_cost = reference_point[1]

        for chromosome in reversed(front):

            width = reference_point[0] - chromosome.makespan

            height = previous_cost - chromosome.total_cost

            if width > 0.0 and height > 0.0:

                hv += width * height

            previous_cost = chromosome.total_cost

        return hv

    @staticmethod
    def spacing(
        pareto_front: list[Chromosome],
    ) -> float:
        """
        Compute Spacing metric.

        Smaller is better.
        """

        if len(pareto_front) < 2:
            return 0.0

        distances: list[float] = []

        for i, first in enumerate(pareto_front):

            nearest = float("inf")

            for j, second in enumerate(pareto_front):

                if i == j:
                    continue

                distance = (
                    abs(first.makespan - second.makespan)
                    + abs(first.total_cost - second.total_cost)
                    + abs(first.total_carbon - second.total_carbon)
                    + abs(first.total_energy - second.total_energy)
                )

                if distance < nearest:
                    nearest = distance

            distances.append(nearest)

        mean_distance = sum(distances) / len(distances)

        variance = sum((distance - mean_distance) ** 2 for distance in distances) / len(
            distances
        )

        return variance**0.5

    @staticmethod
    def default_reference_point(
        pareto_front: list[Chromosome],
    ) -> tuple[float, float]:
        """
        Create a conservative reference point.
        """

        worst_makespan = max(c.makespan for c in pareto_front)

        worst_cost = max(c.total_cost for c in pareto_front)

        return (
            worst_makespan * 1.10,
            worst_cost * 1.10,
        )

    @staticmethod
    def generational_distance(
        pareto_front: list[Chromosome],
        reference_front: list[tuple[float, float]],
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def inverted_generational_distance(
        pareto_front: list[Chromosome],
        reference_front: list[tuple[float, float]],
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def spread(
        pareto_front: list[Chromosome],
    ) -> float:
        raise NotImplementedError
