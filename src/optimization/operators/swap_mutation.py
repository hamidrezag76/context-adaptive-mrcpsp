"""
swap_mutation.py

Swap Mutation Operator

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class SwapMutation:
    """
    Swap two activities inside the priority list.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def mutate(
        self,
        chromosome: Chromosome,
    ) -> None:

        priority = chromosome.priority_list

        if len(priority) < 2:
            return

        i, j = self.random.sample(
            range(len(priority)),
            2,
        )

        priority[i], priority[j] = priority[j], priority[i]
