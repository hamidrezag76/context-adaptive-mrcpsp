from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def main():

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    algorithm = NSGA2(
        project=project,
        population_size=10,
        generations=5,
        seed=1,
        context_adaptive=True,
    )

    population = algorithm.run()

    print()
    print("========== CA-NSGA-II TEST ==========")

    print(
        "Population size:",
        len(population.individuals),
    )

    print(
        "Final context:",
        algorithm.context.get(),
    )

    print(
        "Crossover probability:",
        algorithm.crossover.probability,
    )

    print(
        "Mutation probability:",
        algorithm.mutation.probability,
    )

    print()
    print("Population:")

    for i, chromosome in enumerate(
        population.individuals,
        start=1,
    ):

        print(
            i,
            "| makespan=",
            chromosome.makespan,
            "| cost=",
            round(chromosome.total_cost, 2),
            "| carbon=",
            round(chromosome.total_carbon, 2),
            "| energy=",
            round(chromosome.total_energy, 2),
        )


if __name__ == "__main__":
    main()
