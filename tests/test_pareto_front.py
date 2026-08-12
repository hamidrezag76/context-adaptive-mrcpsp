from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2
from src.optimization.nsga2.fast_non_dominated_sort import (
    FastNonDominatedSort,
)


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    algorithm = NSGA2(
        project,
        population_size=20,
        generations=5,
        seed=42,
    )

    population = algorithm.run()

    sorter = FastNonDominatedSort()

    fronts = sorter.sort(
        population.individuals
    )

    print("\n========== PARETO FRONT ==========")

    print(f"Population : {len(population)}")
    print(f"Number of Fronts : {len(fronts)}")

    for i, front in enumerate(fronts):

        print(
            f"Front {i+1} : {len(front)} solutions"
        )

    print("\n========== FIRST FRONT ==========")

    for chromosome in fronts[0]:

        print(
            chromosome.makespan,
            chromosome.total_cost,
            chromosome.total_carbon,
            chromosome.total_energy,
        )


if __name__ == "__main__":
    main()