from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.decoder import Decoder
from src.optimization.population_initializer import PopulationInitializer
from src.evaluation.evaluator import Evaluator
from src.context.context_manager import ContextManager


def main():

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    population = PopulationInitializer(
        project,
        seed=1,
    ).initialize(10)

    decoder = Decoder(project)
    evaluator = Evaluator(project)

    # --------------------------------------------------
    # Evaluate population
    # --------------------------------------------------

    for chromosome in population.individuals:

        decoded = decoder.decode(
            chromosome
        )

        result = evaluator.evaluate(
            decoded
        )

        chromosome.decoded_schedule = decoded

        chromosome.makespan = (
            result.makespan
        )

        chromosome.total_cost = (
            result.total_cost
        )

        chromosome.total_carbon = (
            result.total_carbon
        )

        chromosome.total_energy = (
            result.total_energy
        )

    # --------------------------------------------------
    # Context
    # --------------------------------------------------

    context_manager = ContextManager(
        project
    )

    context = context_manager.update(
        population,
        generation=1,
        max_generations=80,
    )

    print()
    print("========== CONTEXT TEST ==========")

    print(
        "Context:",
        context,
    )

    print(
        "Vector:",
        context.as_vector(),
    )

    print(
        "All in [0,1]:",
        all(
            0.0 <= x <= 1.0
            for x in context.as_vector()
        ),
    )

    print()

    print("========== POPULATION ==========")

    for i, chromosome in enumerate(
        population.individuals,
        start=1,
    ):

        print(
            i,
            "| makespan=",
            chromosome.makespan,
            "| cost=",
            round(
                chromosome.total_cost,
                2,
            ),
            "| carbon=",
            round(
                chromosome.total_carbon,
                2,
            ),
            "| energy=",
            round(
                chromosome.total_energy,
                2,
            ),
        )


if __name__ == "__main__":

    main()
