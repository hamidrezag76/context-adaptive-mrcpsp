"""
context_analyzer.py

Context Analyzer.

Computes the optimization context used by
the Context-Adaptive NSGA-II.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

import statistics

from src.context.context_vector import ContextVector
from src.models.project import Project
from src.optimization.chromosome import Chromosome


class ContextAnalyzer:
    """
    Analyze optimization state.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:

        self.project = project

    def analyze(
        self,
        population: list[Chromosome],
        generation: int,
        max_generations: int,
    ) -> ContextVector:
        """
        Compute optimization context.
        """

        progress = generation / max_generations

        makespans = [c.makespan for c in population if c.feasible]

        costs = [c.total_cost for c in population if c.feasible]

        carbons = [c.total_carbon for c in population if c.feasible]

        energies = [c.total_energy for c in population if c.feasible]

        diversity = self._population_diversity(
            population,
        )

        return ContextVector(
            schedule_pressure=self._normalize(
                makespans,
            ),
            resource_pressure=0.50,
            cost_pressure=self._normalize(
                costs,
            ),
            carbon_pressure=self._normalize(
                carbons,
            ),
            energy_pressure=self._normalize(
                energies,
            ),
            generation_progress=progress,
            population_diversity=diversity,
        )

    @staticmethod
    def _normalize(
        values: list[float],
    ) -> float:

        if len(values) == 0:
            return 0.0

        minimum = min(values)

        maximum = max(values)

        if maximum == minimum:
            return 0.0

        mean = statistics.mean(values)

        return (mean - minimum) / (maximum - minimum)

    @staticmethod
    def _population_diversity(
        population: list[Chromosome],
    ) -> float:

        if len(population) <= 1:
            return 0.0

        values = []

        for chromosome in population:

            values.append(
                [
                    chromosome.makespan,
                    chromosome.total_cost,
                    chromosome.total_carbon,
                    chromosome.total_energy,
                ]
            )

        import numpy as np

        values = np.array(values, dtype=float)

        minimum = values.min(axis=0)
        maximum = values.max(axis=0)

        denominator = maximum - minimum

        denominator[denominator == 0] = 1.0

        normalized = (values - minimum) / denominator

        centroid = normalized.mean(axis=0)

        distances = np.linalg.norm(
            normalized - centroid,
            axis=1,
        )

        return float(
            distances.mean()
        )
