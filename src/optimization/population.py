"""
population.py

Population container.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.optimization.chromosome import Chromosome

from collections.abc import Callable

from typing import Any

import random


@dataclass(slots=True)
class Population:

    individuals: list[Chromosome] = field(default_factory=list)

    # ----------------------------------------

    def add(
        self,
        chromosome: Chromosome,
    ) -> None:

        self.individuals.append(

            chromosome

        )

    # ----------------------------------------

    def extend(
        self,
        chromosomes: list[Chromosome],
    ) -> None:

        self.individuals.extend(

            chromosomes

        )

    # ----------------------------------------

    def clear(
        self,
    ) -> None:

        self.individuals.clear()

    # ----------------------------------------

    def __len__(

        self,

    ):

        return len(

            self.individuals

        )

    # ----------------------------------------

    def __iter__(

        self,

    ):

        return iter(

            self.individuals

        )

    # ----------------------------------------
    
    def sort(
        self,
        key: Callable[[Chromosome], Any],
        reverse: bool = False,
    ) -> None:

        self.individuals.sort(

            key=key,

            reverse=reverse,

        )
            # ----------------------------------------

    def random_individual(
        self,
    ) -> Chromosome:
        """
        Return one random chromosome.
        """

        if not self.individuals:
            raise ValueError("Population is empty.")

        return random.choice(self.individuals)

    # ----------------------------------------

    def copy(self) -> "Population":
        """
        Deep copy population.
        """

        new_population = Population()

        for chromosome in self.individuals:
            new_population.add(chromosome.copy())

        return new_population

    # ----------------------------------------

    def best(self) -> Chromosome:
        """
        Return best chromosome according to rank and crowding.
        """
        if not self.individuals:

            raise ValueError(
                "Population is empty."
            )

        return sorted(
            self.individuals,
            key=lambda c: (
                c.rank,
                -c.crowding_distance,
            ),
        )[0]
        
    def __getitem__(
        self,
        index: int,
    ) -> Chromosome:
        return self.individuals[index]
    
    @property
    def size(
        self,
    ) -> int:

        return len(self.individuals)
