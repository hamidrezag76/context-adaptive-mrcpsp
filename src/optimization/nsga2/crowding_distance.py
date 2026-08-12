from __future__ import annotations

from src.optimization.chromosome import Chromosome


class CrowdingDistance:
    
    def assign(
            self,
            front: list[Chromosome],
        ) -> None:
    
            self.compute(front)

    def compute(self, front: list[Chromosome]) -> None:

        if len(front) == 0:
            return

        if len(front) <= 2:

            for c in front:
                c.crowding_distance = float("inf")

            return

        objectives = len(front[0].objectives)

        for c in front:
            c.crowding_distance = 0.0

        for m in range(objectives):

            front.sort(
                key=lambda x: x.objectives[m]
            )

            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            fmin = front[0].objectives[m]
            fmax = front[-1].objectives[m]

            if fmax == fmin:
                continue

            for i in range(1, len(front)-1):

                prev = front[i-1].objectives[m]
                nxt = front[i+1].objectives[m]

                front[i].crowding_distance += (
                    (nxt-prev)/(fmax-fmin)
                )
                
    
