from __future__ import annotations

import random

from src.models.project import Project
from src.optimization.chromosome import Chromosome


class Mutation:
    """
    Mutation operators for CA-NSGA-II.

    Supports

        1. Priority Swap Mutation

        2. Mode Mutation

        3. Adaptive Mutation Rate
    """

    def __init__(

        self,

        project: Project,

        probability: float = 0.15,

        seed: int | None = None,

    ) -> None:

        self.project = project

        self.probability = probability

        self.random = random.Random(seed)

    # =====================================================
    # TODO:
    # Replace fixed mutation probability with
    # adaptive_probability(context_factor)
    # when Context Controller is integrated.

    def mutate(

        self,

        chromosome: Chromosome,

    ) -> Chromosome:

        child = chromosome.copy()

        if self.random.random() < self.probability:

            self.priority_swap(child)

        if self.random.random() < self.probability:

            self.mode_mutation(child)
            
            assert len(set(child.priority_list)) == len(
                child.priority_list
            )

        return child
    
    def apply(
        self,
        chromosome: Chromosome,
    ) -> Chromosome:
        """
        Public API.
        """
        return self.mutate(chromosome)

    # =====================================================

    def priority_swap(

        self,

        chromosome: Chromosome,

    ) -> None:

        n = len(chromosome.priority_list)

        if n < 2:

            return

        i, j = self.random.sample(
            range(n),
            2,
        )

        chromosome.priority_list[i], chromosome.priority_list[j] = (

            chromosome.priority_list[j],

            chromosome.priority_list[i],

        )
        
        assert len(set(chromosome.priority_list)) == len(
            chromosome.priority_list
        )

    # =====================================================

    def mode_mutation(

        self,

        chromosome: Chromosome,

    ) -> None:

        activity_id = self.random.choice(

            list(chromosome.mode_assignment.keys())

        )

        activity = self.project.activities[activity_id]

        if len(activity.modes) <= 1:

            return

        current = chromosome.mode_assignment[activity_id]
        old_mode = current

        candidates = [

            m.id

            for m in activity.modes

            if m.id != current

        ]
        
        if not candidates:

            return

        chromosome.mode_assignment[activity_id] = self.random.choice(

            candidates

        )

    # =====================================================

    def adaptive_probability(

        self,

        context_factor: float,

    ) -> float:

        """
        Context-aware mutation rate.

        context_factor

            0.0 = stable

            1.0 = highly dynamic
        """

        return min(

            0.40,

            self.probability * (1.0 + context_factor),

        )