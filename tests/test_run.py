from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    algorithm = NSGA2(

        project,

        population_size=10,

        generations=5,

        seed=42,

    )

    population = algorithm.run()

    print()

    print("Final population:", len(population))


if __name__ == "__main__":

    main()