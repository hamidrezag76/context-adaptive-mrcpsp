"""
fast_non_dominated_sort.py

Fast Non-Dominated Sorting
(Deb et al., 2002)

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from typing import List

from src.optimization.chromosome import Chromosome


class FastNonDominatedSort:
    """
    Fast Non-Dominated Sorting.

    Assigns Pareto ranks to a population.
    """

    # ---------------------------------------------------------

    @staticmethod
    def dominates(a: Chromosome, b: Chromosome) -> bool:
        """
        True if chromosome a dominates chromosome b.
        """

        obj_a = a.objectives
        obj_b = b.objectives

        better_or_equal = True
        strictly_better = False

        for x, y in zip(obj_a, obj_b):

            if x > y:
                better_or_equal = False
                break

            if x < y:
                strictly_better = True

        return better_or_equal and strictly_better

    # ---------------------------------------------------------

    def sort(
        self,
        population: List[Chromosome],
    ) -> List[List[Chromosome]]:
        """
        Returns all Pareto fronts.
        """

        fronts: List[List[Chromosome]] = [[]]

        for p in population:

            p.dominated_solutions = []

            p.domination_count = 0

        # -----------------------------------------

        for p in population:

            for q in population:

                if p is q:
                    continue

                if self.dominates(p, q):

                    p.dominated_solutions.append(q)

                elif self.dominates(q, p):

                    p.domination_count += 1

            if p.domination_count == 0:

                p.rank = 1

                fronts[0].append(p)

        # -----------------------------------------

        i = 0

        while i < len(fronts):

            next_front = []

            for p in fronts[i]:

                for q in p.dominated_solutions:

                    q.domination_count -= 1

                    if q.domination_count == 0:

                        q.rank = i + 2

                        next_front.append(q)

            if next_front:

                fronts.append(next_front)

            i += 1

        return fronts
