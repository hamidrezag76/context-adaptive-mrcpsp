from pathlib import Path

from src.parser.mm_parser import MMParser

from src.optimization.ca_nsga2 import NSGA2


def main():

    parser = MMParser(

        Path("benchmarks/data/j301_1.mm")

    )

    project = parser.parse()

    algorithm = NSGA2(

        project,

        population_size=20,

        generations=10,

        seed=42,

    )

    algorithm.prepare()

    print(

        len(algorithm.population.individuals)

    )

    print(

        algorithm.population.individuals[0].rank

    )


if __name__ == "__main__":

    main()