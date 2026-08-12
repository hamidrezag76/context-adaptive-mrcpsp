from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.population_initializer import PopulationInitializer


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    initializer = PopulationInitializer(project)

    population = initializer.initialize(20)

    print(len(population))

    chromosome = population.random()

    print(chromosome.priority_list[:10])

    print(
        list(chromosome.mode_assignment.items())[:10]
    )


if __name__ == "__main__":

    main()