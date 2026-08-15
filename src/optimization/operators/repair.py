from __future__ import annotations

from src.models.project import Project
from src.optimization.chromosome import Chromosome


class Repair:
    """
    Repairs invalid chromosomes.
    """

    def __init__(self, project: Project):

        self.project = project

    # ---------------------------------------------------------

    def repair(
        self,
        chromosome: Chromosome,
    ) -> Chromosome:

        self._repair_priority(chromosome)

        self._repair_modes(chromosome)

        assert len(chromosome.priority_list) == len(
            self.project.activities
        )

        assert len(chromosome.mode_assignment) == len(
            self.project.activities
        )

        return chromosome

    # ---------------------------------------------------------
    # TODO:
    # Implement precedence-feasible repair.
    # Current implementation only removes duplicates
    # and restores missing activities.

    def _repair_priority(
        self,
        chromosome: Chromosome,
    ) -> None:

        valid = sorted(self.project.activities.keys())

        seen = set()

        repaired = []

        for a in chromosome.priority_list:

            if a in valid and a not in seen:

                repaired.append(a)

                seen.add(a)

        for a in valid:

            if a not in seen:

                repaired.append(a)

        chromosome.priority_list = repaired
        
        assert len(repaired) == len(valid)

        assert len(set(repaired)) == len(valid)

    # ---------------------------------------------------------

    def _repair_modes(
        self,
        chromosome: Chromosome,
    ) -> None:

        capacities = (
            self.project.renewable_capacities
        )

        for activity in (
            self.project.activities.values()
        ):

            if not activity.modes:

                raise ValueError(
                    f"Activity {activity.id} "
                    "has no execution modes."
                )

            # -------------------------------------------------
            # Determine resource-feasible modes
            # -------------------------------------------------

            feasible_modes = [
                mode
                for mode in activity.modes
                if len(mode.renewable)
                <= len(capacities)
                and all(
                    requirement <= capacity
                    for requirement, capacity
                    in zip(
                        mode.renewable,
                        capacities,
                    )
                )
            ]

            if not feasible_modes:

                raise ValueError(
                    "Activity has no resource-feasible "
                    f"execution mode: "
                    f"activity={activity.id}"
                )

            # -------------------------------------------------
            # Missing assignment
            # -------------------------------------------------

            if activity.id not in (
                chromosome.mode_assignment
            ):

                chromosome.mode_assignment[
                    activity.id
                ] = feasible_modes[0].id

                continue

            # -------------------------------------------------
            # Existing assignment
            # -------------------------------------------------

            mode_id = chromosome.mode_assignment[
                activity.id
            ]

            valid_mode_ids = {
                mode.id
                for mode in activity.modes
            }

            # Invalid mode ID
            if mode_id not in valid_mode_ids:

                chromosome.mode_assignment[
                    activity.id
                ] = feasible_modes[0].id

                continue

            # -------------------------------------------------
            # Existing mode is resource-infeasible
            # -------------------------------------------------

            selected_mode = next(
                mode
                for mode in activity.modes
                if mode.id == mode_id
            )

            selected_feasible = all(
                requirement <= capacity
                for requirement, capacity
                in zip(
                    selected_mode.renewable,
                    capacities,
                )
            )

            if not selected_feasible:

                chromosome.mode_assignment[
                    activity.id
                ] = feasible_modes[0].id
    
    def apply(
        self,
        chromosome: Chromosome,
    ) -> Chromosome:
        """
        Public API.
        """
        return self.repair(chromosome)