"""
insertion_mutation.py

Insertion Mutation Operator

Moves one activity to another position while
preserving permutation feasibility.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class InsertionMutation:
    """
    Insertion mutation operator.

    Removes one activity from the priority list and inserts it
    into another position.
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
        """
        Apply insertion mutation in-place.
        """

        n = len(chromosome.priority_list)

        if n <= 2:
            return

        source = self.random.randrange(n)

        destination = self.random.randrange(n)

        while destination == source:
            destination = self.random.randrange(n)

        gene = chromosome.priority_list.pop(source)

        chromosome.priority_list.insert(
            destination,
            gene,
        )
