from __future__ import annotations

import random

from src.optimization.chromosome import Chromosome


class Crossover:
    """
    Genetic crossover operators.

    Implements

        1. Priority Order Crossover (POX)

        2. Mode Crossover
    """

    def __init__(

        self,

        probability: float = 0.90,

        seed: int | None = None,

    ) -> None:

        self.probability = probability

        self.random = random.Random(seed)

    # =====================================================

    def crossover(

        self,

        parent1: Chromosome,

        parent2: Chromosome,

    ) -> tuple[Chromosome, Chromosome]:

        """
        Perform crossover.
        """

        if self.random.random() > self.probability:

            return (

                parent1.copy(),

                parent2.copy(),

            )

        child1_priority, child2_priority = self._priority_crossover(

            parent1.priority_list,

            parent2.priority_list,

        )

        child1_modes, child2_modes = self._mode_crossover(

            parent1.mode_assignment,

            parent2.mode_assignment,

        )

        child1 = Chromosome(

            priority_list=child1_priority,

            mode_assignment=child1_modes,

        )

        child2 = Chromosome(

            priority_list=child2_priority,

            mode_assignment=child2_modes,

        )

        return child1, child2

    # =====================================================

    def _priority_crossover(

        self,

        p1: list[int],

        p2: list[int],

    ) -> tuple[list[int], list[int]]:

        """
        Order Crossover (OX)

        Preserves permutation validity.
        """

        n = len(p1)
        
        if len(p1) != len(p2):

            raise ValueError(
                "Parents must have equal length."
            )

        cut1 = self.random.randint(0, n - 2)

        cut2 = self.random.randint(cut1 + 1, n - 1)

        child1 = [-1] * n

        child2 = [-1] * n

        child1[cut1:cut2] = p1[cut1:cut2]

        child2[cut1:cut2] = p2[cut1:cut2]

        used1 = set(child1)
        used1.discard(-1)

        used2 = set(child2)
        used2.discard(-1)

        fill1 = [

            g

            for g in p2

            if g not in used1

        ]

        fill2 = [

            g

            for g in p1

            if g not in used2

        ]

        idx = 0

        for i in range(n):

            if child1[i] == -1:

                child1[i] = fill1[idx]

                idx += 1

        idx = 0

        for i in range(n):

            if child2[i] == -1:

                child2[i] = fill2[idx]

                idx += 1

        # Validation
        assert len(set(child1)) == len(child1)
        assert len(set(child2)) == len(child2)

        return child1, child2

    # =====================================================

    def _mode_crossover(

        self,

        m1: dict[int, int],

        m2: dict[int, int],

    ) -> tuple[dict[int, int], dict[int, int]]:

        """
        Uniform crossover for execution modes.
        """

        child1 = {}

        child2 = {}

        for activity in sorted(m1):

            if self.random.random() < 0.5:

                child1[activity] = m1[activity]

                child2[activity] = m2[activity]

            else:

                child1[activity] = m2[activity]

                child2[activity] = m1[activity]

        return child1, child2