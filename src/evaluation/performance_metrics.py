"""
performance_metrics.py

Performance Metrics for Multi-Objective Optimization

Author
------
CA-SMRCPSP Research Project
"""
from __future__ import annotations

import math
import numpy as np

from pymoo.indicators.hv import HV

from src.optimization.chromosome import Chromosome

import math

import numpy as np

from src.optimization.chromosome import Chromosome


class PerformanceMetrics:
    """
    Computes performance indicators for Pareto fronts.
    """

    @staticmethod
    def number_of_solutions(
        pareto_front: list[Chromosome],
    ) -> int:
        return len(pareto_front)

    @staticmethod
    def spacing(
        front: np.ndarray,
    ) -> float:
        """
        Spacing metric on a normalized objective front.

        Lower values indicate a more uniformly distributed
        Pareto front.
        """

        if len(front) < 2:
            return 0.0

        front = np.asarray(
            front,
            dtype=float,
        )

        nearest_distances = []

        for i in range(len(front)):

            distances = np.linalg.norm(
                front[i] - np.delete(
                    front,
                    i,
                    axis=0,
                ),
                axis=1,
            )

            nearest_distances.append(
                np.min(distances)
            )

        nearest_distances = np.asarray(
            nearest_distances,
            dtype=float,
        )

        mean_distance = np.mean(
            nearest_distances
        )

        if len(nearest_distances) == 1:
            return 0.0

        return float(
            np.sqrt(
                np.sum(
                    (
                        nearest_distances
                        - mean_distance
                    ) ** 2
                )
                / (
                    len(nearest_distances) - 1
                )
            )
        )

    @staticmethod
    def spread(
        pareto_front: list[Chromosome],
    ) -> float:

        if len(pareto_front) < 3:
            return 0.0

        front = sorted(
            pareto_front,
            key=lambda c: c.makespan,
        )

        distances = []

        for i in range(len(front) - 1):

            d = (
                abs(front[i].makespan - front[i + 1].makespan)
                + abs(front[i].total_cost - front[i + 1].total_cost)
                + abs(front[i].total_carbon - front[i + 1].total_carbon)
                + abs(front[i].total_energy - front[i + 1].total_energy)
            )

            distances.append(d)

        mean = np.mean(distances)

        delta = np.sum(np.abs(np.array(distances) - mean))

        return float(delta / ((len(distances)) * mean))

    @staticmethod
    def generational_distance(
        obtained: np.ndarray,
        reference: np.ndarray,
    ) -> float:

        if len(obtained) == 0:
            return math.inf

        distances = []

        for point in obtained:

            d = np.min(np.linalg.norm(reference - point, axis=1))

            distances.append(d)

        return float(np.mean(distances))

    @staticmethod
    def inverted_generational_distance(
        obtained,
        reference,
    ):

        if len(obtained) == 0:
            return float("inf")

        distances = []

        for ref in reference:

            nearest = np.min(
                np.linalg.norm(
                    obtained - ref,
                    axis=1,
                )
            )

            distances.append(nearest)

        return float(
            np.mean(
                distances
            )
        )

    @staticmethod
    def hypervolume(
        front: np.ndarray,
        reference_point: np.ndarray,
    ) -> float:

        if len(front) == 0:
            return 0.0

        # ==========================
        # DEBUG
        # ==========================

        assert np.all(front >= 0), "Found value smaller than 0"

        assert np.all(front <= 1.000001), "Found value larger than 1"

        hv = HV(
            ref_point=reference_point,
        )

        value = float(hv(front))
        return value

    @staticmethod
    def summary(
        pareto_front: list[Chromosome],
    ) -> dict[str, float]:
        """
        Compute a summary of the Pareto front performance.
        """

        if not pareto_front:
            return {
                "solutions": 0,
                "spacing": 0.0,
                "spread": 0.0,
            }

        return {
            "solutions": PerformanceMetrics.number_of_solutions(
                pareto_front,
            ),
            "spacing": round(
                PerformanceMetrics.spacing(
                    pareto_front,
                ),
                4,
            ),
            "spread": round(
                PerformanceMetrics.spread(
                    pareto_front,
                ),
                4,
            ),
        }

    @staticmethod
    def normalize_pair(
        obtained: np.ndarray,
        reference: np.ndarray,
    ):
        """
        Normalize both fronts using the SAME bounds.
        """

        combined = np.vstack(
            (
                obtained,
                reference,
            )
        )

        minimum = combined.min(axis=0)
        maximum = combined.max(axis=0)

        denominator = maximum - minimum
        denominator[denominator == 0.0] = 1.0

        obtained_norm = (
            obtained - minimum
        ) / denominator

        reference_norm = (
            reference - minimum
        ) / denominator

        return (
            obtained_norm,
            reference_norm,
        )
    
