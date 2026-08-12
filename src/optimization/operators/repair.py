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

        for activity in self.project.activities.values():

            if activity.id not in chromosome.mode_assignment:
                
                if not activity.modes:

                    raise ValueError(
                        f"Activity {activity.id} has no execution modes."
                    )

                chromosome.mode_assignment[activity.id] = activity.modes[0].id

                continue

            mode_id = chromosome.mode_assignment[activity.id]

            valid = {m.id for m in activity.modes}

            if mode_id not in valid:
                
                if not activity.modes:

                    raise ValueError(
                        f"Activity {activity.id} has no execution modes."
                    )

                chromosome.mode_assignment[activity.id] = activity.modes[0].id
    
    def apply(
        self,
        chromosome: Chromosome,
    ) -> Chromosome:
        """
        Public API.
        """
        return self.repair(chromosome)