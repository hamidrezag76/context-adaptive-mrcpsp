"""
order_crossover.py

Order Crossover (OX)

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class OrderCrossover:
    """
    Standard Order Crossover (OX).
    """

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

    def crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:

        size = len(parent1.priority_list)

        left = self.random.randint(0, size - 2)
        right = self.random.randint(left + 1, size - 1)

        child1 = self._ox(
            parent1.priority_list,
            parent2.priority_list,
            left,
            right,
        )

        child2 = self._ox(
            parent2.priority_list,
            parent1.priority_list,
            left,
            right,
        )

        return (
            Chromosome(
                priority_list=child1,
                mode_assignment=parent1.mode_assignment.copy(),
            ),
            Chromosome(
                priority_list=child2,
                mode_assignment=parent2.mode_assignment.copy(),
            ),
        )

    @staticmethod
    def _ox(
        p1: list[int],
        p2: list[int],
        left: int,
        right: int,
    ) -> list[int]:

        size = len(p1)

        child = [-1] * size

        child[left : right + 1] = p1[left : right + 1]

        pointer = (right + 1) % size

        for gene in p2:

            if gene in child:
                continue

            while child[pointer] != -1:
                pointer = (pointer + 1) % size

            child[pointer] = gene

        return child
