from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    algorithm = NSGA2(
        project,
        population_size=20,
        generations=5,
    )

    algorithm.prepare()

    print(len(algorithm.population))

    print(algorithm.population[0].rank)

    print(algorithm.population[0].makespan)


if __name__ == "__main__":
    main()