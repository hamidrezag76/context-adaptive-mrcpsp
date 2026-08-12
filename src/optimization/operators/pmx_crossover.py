"""
pmx_crossover.py

Partially Mapped Crossover (PMX)

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class PMXCrossover:

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

        child1 = self._pmx(
            parent1.priority_list,
            parent2.priority_list,
            left,
            right,
        )

        child2 = self._pmx(
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
    def _pmx(
        p1: list[int],
        p2: list[int],
        left: int,
        right: int,
    ) -> list[int]:
        """
        Standard Partially Matched Crossover (PMX).
        """

        size = len(p1)

        child = [-1] * size

        # --------------------------------------------------
        # Step 1
        # Copy crossover segment
        # --------------------------------------------------

        child[left : right + 1] = p1[left : right + 1]

        # --------------------------------------------------
        # Step 2
        # Resolve mapping conflicts
        # --------------------------------------------------

        for i in range(left, right + 1):

            gene = p2[i]

            if gene in child:
                continue

            position = i

            while True:

                mapped_gene = p1[position]

                position = p2.index(mapped_gene)

                if child[position] == -1:

                    child[position] = gene

                    break

        # --------------------------------------------------
        # Step 3
        # Fill remaining positions
        # --------------------------------------------------

        for i in range(size):

            if child[i] == -1:

                child[i] = p2[i]

        return child
