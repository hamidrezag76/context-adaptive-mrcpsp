from __future__ import annotations

import random

from src.models.project import Project
from src.optimization.chromosome import Chromosome
from src.optimization.population import Population


class PopulationInitializer:
    """
    Generates an initial feasible population.
    """

    def __init__(
        self,
        project: Project,
        seed: int | None = None,
    ):

        self.project = project

        self.random = random.Random(seed)

    # ---------------------------------------------------------

    def initialize(
        self,
        population_size: int,
    ) -> Population:

        population = Population()

        activity_ids = list(self.project.activities.keys())

        for _ in range(population_size):

            priority = self._random_topological_order()

            mode_assignment = {}

            for activity in self.project.activities.values():

                mode = self.random.choice(activity.modes)

                mode_assignment[activity.id] = mode.id

            chromosome = Chromosome(

                priority_list=priority,

                mode_assignment=mode_assignment,

            )

            population.add(chromosome)

        return population
    
    def _random_topological_order(
        self,
    ) -> list[int]:

        predecessors = {
            activity.id: set(activity.predecessors)
            for activity in self.project.activities.values()
        }

        available = [

            activity_id

            for activity_id, preds in predecessors.items()

            if not preds

        ]

        order = []

        while available:

            activity = self.random.choice(available)

            available.remove(activity)

            order.append(activity)

            for successor in self.project.activities[activity].successors:

                predecessors[successor].remove(activity)

                if not predecessors[successor]:

                    available.append(successor)

        if len(order) != len(self.project.activities):

            raise RuntimeError(
                "Topological ordering failed."
            )

        return order