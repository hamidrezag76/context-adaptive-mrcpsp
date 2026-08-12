from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class TournamentSelection:
    """
    Binary Tournament Selection for NSGA-II.

    Priority:
        1. Lower rank wins.
        2. If equal rank, larger crowding distance wins.
    """

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

    # ---------------------------------------------------------

    def select(
        self,
        population: list[Chromosome],
    ) -> Chromosome:

        if len(population) < 2:

            raise ValueError(
                "Population must contain at least two chromosomes."
            )

        a, b = self.random.sample(
            population,
            2,
        )

        return self._better(a, b)

    # ---------------------------------------------------------

    def select_pair(
        self,
        population: list[Chromosome],
    ) -> tuple[Chromosome, Chromosome]:

       parent1 = self.select(population)

       while True:

            parent2 = self.select(population)

            if parent2 is not parent1:

                break

       return parent1, parent2

    # ---------------------------------------------------------

    def _better(
        self,
        a: Chromosome,
        b: Chromosome,
    ) -> Chromosome:

        # Lower rank dominates

        if a.rank < b.rank:

            return a

        if b.rank < a.rank:

            return b

        # Higher crowding distance

        if a.crowding_distance > b.crowding_distance:

            return a

        if b.crowding_distance > a.crowding_distance:

            return b

        # Random tie-break

        if self.random.random() < 0.5:

            return a

        return b