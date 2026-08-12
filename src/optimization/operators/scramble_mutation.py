"""
scramble_mutation.py

Scramble Mutation Operator

Randomly shuffles one subsequence of the priority list.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class ScrambleMutation:
    """
    Scramble mutation.

    Randomly shuffles one segment.
    """

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

    def mutate(
        self,
        chromosome: Chromosome,
    ) -> None:

        n = len(chromosome.priority_list)

        if n <= 2:
            return

        left = self.random.randint(
            0,
            n - 2,
        )

        right = self.random.randint(
            left + 1,
            n - 1,
        )

        segment = chromosome.priority_list[left : right + 1]

        self.random.shuffle(segment)

        chromosome.priority_list[left : right + 1] = segment
