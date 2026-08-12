"""
Fast Non Dominated Sorting
Deb et al. (2002)

CA-SMRCPSP
"""

from __future__ import annotations

from typing import List

from src.optimization.chromosome import Chromosome


class FastNonDominatedSort:
    """
    Deb Fast Non Dominated Sorting.
    """

    def dominates(
        self,
        p: Chromosome,
        q: Chromosome,
    ) -> bool:
        if len(p.objectives) != len(q.objectives):

            raise ValueError(
                "Objective dimensions do not match."
            )
            
        better = False

        for a, b in zip(
            p.objectives,
            q.objectives,
        ):
            if a > b:
                return False
            if a < b:
                better = True

        return better
    
    def sort(
        self,
        population: List[Chromosome],
    ) -> List[List[Chromosome]]:
        
        if not population:
            return []

        fronts: List[List[Chromosome]] = [[]]

        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []

        for p in population:
            for q in population:
                if p is q:
                    continue

                if self.dominates(
                    p,
                    q,
                ):
                    p.dominated_solutions.append(q)
                elif self.dominates(
                    q,
                    p,
                ):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 1
                fronts[0].append(p)

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

        return [

            front

            for front in fronts

            if front

        ]
