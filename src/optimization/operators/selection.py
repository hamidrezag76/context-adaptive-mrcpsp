from __future__ import annotations

import random

from src.context.context import Context
from src.optimization.chromosome import Chromosome


class TournamentSelection:
    """
    Binary Tournament Selection for NSGA-II.

    Standard mode
    -------------
    Priority:
        1. Lower Pareto rank.
        2. Larger crowding distance.
        3. Random tie-break.

    Context-aware mode
    ------------------
    Priority:
        1. Lower Pareto rank.
        2. Lower context-guided preference score.
        3. Larger crowding distance.
        4. Random tie-break.

    Context-aware selection does not alter Pareto dominance.
    Context is used only as a search-guidance mechanism when
    competing chromosomes have the same Pareto rank.

    All objective values are normalized within the current
    tournament population before the context preference is
    calculated.
    """

    def __init__(
        self,
        seed: int | None = None,
        context_adaptive: bool = False,
    ) -> None:

        self.random = random.Random(seed)

        self.context_adaptive = context_adaptive

    # ---------------------------------------------------------
    # Selection
    # ---------------------------------------------------------

    def select(
        self,
        population: list[Chromosome],
        context: Context | None = None,
    ) -> Chromosome:

        if len(population) < 2:
            raise ValueError(
                "Population must contain at least two chromosomes."
            )

        a, b = self.random.sample(
            population,
            2,
        )

        return self._better(
            a,
            b,
            population=population,
            context=context,
        )

    # ---------------------------------------------------------

    def select_pair(
        self,
        population: list[Chromosome],
        context: Context | None = None,
    ) -> tuple[Chromosome, Chromosome]:

        parent1 = self.select(
            population,
            context=context,
        )

        while True:

            parent2 = self.select(
                population,
                context=context,
            )

            if parent2 is not parent1:
                break

        return parent1, parent2

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    def _better(
        self,
        a: Chromosome,
        b: Chromosome,
        population: list[Chromosome],
        context: Context | None,
    ) -> Chromosome:

        # -----------------------------------------------------
        # Primary NSGA-II criterion: Pareto rank
        # -----------------------------------------------------

        if a.rank < b.rank:
            return a

        if b.rank < a.rank:
            return b

        # -----------------------------------------------------
        # Context-aware preference
        # -----------------------------------------------------

        if (
            self.context_adaptive
            and context is not None
        ):

            score_a = self._context_preference(
                a,
                population,
                context,
            )

            score_b = self._context_preference(
                b,
                population,
                context,
            )

            if score_a < score_b:
                return a

            if score_b < score_a:
                return b

        # -----------------------------------------------------
        # Standard NSGA-II secondary criterion
        # -----------------------------------------------------

        if (
            a.crowding_distance
            > b.crowding_distance
        ):
            return a

        if (
            b.crowding_distance
            > a.crowding_distance
        ):
            return b

        # -----------------------------------------------------
        # Random tie-break
        # -----------------------------------------------------

        if self.random.random() < 0.5:
            return a

        return b

    # ---------------------------------------------------------
    # Context preference
    # ---------------------------------------------------------

    @staticmethod
    def _context_preference(
        chromosome: Chromosome,
        population: list[Chromosome],
        context: Context,
    ) -> float:
        """
        Calculate a normalized context-guided preference score.

        Lower values are preferred.

        Objective normalization is performed over the current
        population to avoid mixing incompatible objective scales.

        Context pressures determine the relative emphasis of
        schedule, cost, carbon, and energy objectives.

        Resource pressure and uncertainty are intentionally not
        mapped directly to an objective because Chromosome does
        not contain independent resource or uncertainty
        objectives.
        """

        objectives = (
            "makespan",
            "total_cost",
            "total_carbon",
            "total_energy",
        )

        weights = (
            context.schedule_pressure,
            context.cost_pressure,
            context.carbon_pressure,
            context.energy_pressure,
        )

        weight_sum = sum(weights)

        if weight_sum <= 0.0:
            weights = (
                1.0,
                1.0,
                1.0,
                1.0,
            )
            weight_sum = 4.0

        normalized = []

        for objective in objectives:

            values = [
                float(
                    getattr(
                        individual,
                        objective,
                    )
                )
                for individual in population
            ]

            minimum = min(values)
            maximum = max(values)

            value = float(
                getattr(
                    chromosome,
                    objective,
                )
            )

            if maximum == minimum:

                normalized_value = 0.0

            else:

                normalized_value = (
                    (value - minimum)
                    / (maximum - minimum)
                )

            normalized.append(
                normalized_value
            )

        return float(
            sum(
                weight * value
                for weight, value
                in zip(
                    weights,
                    normalized,
                )
            )
            / weight_sum
        )
