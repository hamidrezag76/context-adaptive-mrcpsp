"""
archive.py

External Elite Archive
"""

from __future__ import annotations

from src.optimization.chromosome import Chromosome


class EliteArchive:

    def __init__(
        self,
        maximum_size: int = 100,
    ):

        self.maximum_size = maximum_size

        self.archive: list[Chromosome] = []
        
        self.current_context = None

    @staticmethod
    def dominates(
        a: Chromosome,
        b: Chromosome,
    ) -> bool:

        objectives_a = (
            a.makespan,
            a.total_cost,
            a.total_carbon,
            a.total_energy,
        )

        objectives_b = (
            b.makespan,
            b.total_cost,
            b.total_carbon,
            b.total_energy,
        )

        better_or_equal = all(
            x <= y
            for x, y in zip(
                objectives_a,
                objectives_b,
            )
        )

        strictly_better = any(
            x < y
            for x, y in zip(
                objectives_a,
                objectives_b,
            )
        )

        return better_or_equal and strictly_better
    
    def set_context(
        self,
        context,
    ) -> None:
        """
        Receive current optimization context.
        """
        self.current_context = context

    def update(
        self,
        solutions: list[Chromosome],
    ) -> None:

        pool = self.archive + solutions

        new_archive = []

        for solution in pool:

            dominated = False

            for other in pool:

                if solution is other:
                    continue

                if self.dominates(
                    other,
                    solution,
                ):
                    dominated = True
                    break

            if not dominated:
                new_archive.append(solution)

        unique = {}

        for chromosome in new_archive:

            key = (
                chromosome.makespan,
                chromosome.total_cost,
                chromosome.total_carbon,
                chromosome.total_energy,
            )

            unique[key] = chromosome

        self.archive = list(unique.values())

        self.archive.sort(
            key=lambda c: (
                c.makespan,
                c.total_cost,
                c.total_carbon,
                c.total_energy,
            )
        )

        if len(self.archive) > self.maximum_size:

            # --------------------------------------------------
            # Capacity control using crowding distance
            # --------------------------------------------------

            if len(self.archive) > self.maximum_size:

                front = self.archive

                for chromosome in front:
                    chromosome.crowding_distance = 0.0

                objectives = [
                    "makespan",
                    "total_cost",
                    "total_carbon",
                    "total_energy",
                ]

                for objective in objectives:

                    front.sort(
                        key=lambda c: getattr(c, objective),
                    )

                    front[0].crowding_distance = float("inf")
                    front[-1].crowding_distance = float("inf")

                    minimum = getattr(front[0], objective)
                    maximum = getattr(front[-1], objective)

                    if maximum == minimum:
                        continue

                    for i in range(1, len(front) - 1):

                        previous = getattr(front[i - 1], objective)
                        following = getattr(front[i + 1], objective)

                        front[i].crowding_distance += (
                            following - previous
                        ) / (maximum - minimum)

                if self.current_context is None:

                    front.sort(
                        key=lambda c: c.crowding_distance,
                        reverse=True,
                    )

                else:

                    context = self.current_context

                    front.sort(
                        key=lambda c: (
                            c.crowding_distance
                            + 0.30 * (
                                context.schedule_pressure
                                * (
                                    c.makespan
                                    / max(
                                        1,
                                        front[-1].makespan,
                                    )
                                )
                            )
                            + 0.25 * (
                                context.cost_pressure
                                * (
                                    c.total_cost
                                    / max(
                                        1,
                                        front[-1].total_cost,
                                    )
                                )
                            )
                            + 0.25 * (
                                context.carbon_pressure
                                * (
                                    c.total_carbon
                                    / max(
                                        1,
                                        front[-1].total_carbon,
                                    )
                                )
                            )
                            + 0.20 * (
                                context.energy_pressure
                                * (
                                    c.total_energy
                                    / max(
                                        1,
                                        front[-1].total_energy,
                                    )
                                )
                            )
                        ),
                        reverse=True,
                    )

                self.archive = front[: self.maximum_size]
