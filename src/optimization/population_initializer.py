from __future__ import annotations

import random

from src.models.project import Project
from src.optimization.chromosome import Chromosome
from src.optimization.population import Population


class PopulationInitializer:
    """
    Generates an initial population with
    resource-feasible execution modes.
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

        if population_size <= 0:

            raise ValueError(
                "Population size must be positive."
            )

        population = Population()

        activity_ids = list(
            self.project.activities.keys()
        )

        for _ in range(population_size):

            priority = (
                self._random_topological_order()
            )

            mode_assignment = {}

            for activity_id in activity_ids:

                activity = self.project.activities[
                    activity_id
                ]

                feasible_modes = (
                    self._feasible_modes(activity)
                )

                if not feasible_modes:

                    raise ValueError(
                        "Activity has no resource-feasible "
                        f"execution mode: "
                        f"activity={activity.id}"
                    )

                mode = self.random.choice(
                    feasible_modes
                )

                mode_assignment[
                    activity.id
                ] = mode.id

            chromosome = Chromosome(
                priority_list=priority,
                mode_assignment=mode_assignment,
            )

            population.add(
                chromosome
            )

        return population

    # ---------------------------------------------------------

    def _feasible_modes(
        self,
        activity,
    ):

        """
        Return execution modes whose renewable
        resource requirements do not exceed project
        renewable capacities.
        """

        capacities = (
            self.project.renewable_capacities
        )

        feasible = []

        for mode in activity.modes:

            if len(mode.renewable) > len(capacities):

                continue

            if all(
                requirement <= capacity
                for requirement, capacity
                in zip(
                    mode.renewable,
                    capacities,
                )
            ):

                feasible.append(
                    mode
                )

        return feasible

    # ---------------------------------------------------------

    def _random_topological_order(
        self,
    ) -> list[int]:

        predecessors = {
            activity.id: set(
                activity.predecessors
            )
            for activity
            in self.project.activities.values()
        }

        available = [
            activity_id
            for activity_id, preds
            in predecessors.items()
            if not preds
        ]

        order = []

        while available:

            activity = self.random.choice(
                available
            )

            available.remove(
                activity
            )

            order.append(
                activity
            )

            for successor in (
                self.project.activities[
                    activity
                ].successors
            ):

                predecessors[
                    successor
                ].remove(
                    activity
                )

                if not predecessors[
                    successor
                ]:

                    available.append(
                        successor
                    )

        if len(order) != len(
            self.project.activities
        ):

            raise RuntimeError(
                "Topological ordering failed."
            )

        return order