"""
local_search.py

Context-Adaptive Local Search
"""

from __future__ import annotations

from copy import deepcopy

from src.optimization.chromosome import Chromosome


class LocalSearch:

    def __init__(self, decoder):
        self.decoder = decoder

    def improve(
        self,
        chromosome: Chromosome,
        iterations: int = 10,
    ) -> Chromosome:

        best = deepcopy(chromosome)

        best_result = self.decoder.decode_and_evaluate(best)

        best.makespan = best_result.makespan
        best.total_cost = best_result.total_cost
        best.total_carbon = best_result.total_carbon
        best.total_energy = best_result.total_energy

        for _ in range(iterations):

            candidate = deepcopy(best)

            i = candidate.priority_list.index(
                min(candidate.priority_list)
            )

            j = candidate.priority_list.index(
                max(candidate.priority_list)
            )

            candidate.priority_list[i], candidate.priority_list[j] = (
                candidate.priority_list[j],
                candidate.priority_list[i],
            )

            result = self.decoder.decode_and_evaluate(candidate)

            candidate.makespan = result.makespan
            candidate.total_cost = result.total_cost
            candidate.total_carbon = result.total_carbon
            candidate.total_energy = result.total_energy

            if (
                candidate.makespan
                <= best.makespan
                and candidate.total_cost
                <= best.total_cost
            ):
                best = candidate

        return best
