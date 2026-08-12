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
        seed=42,
    )

    population = algorithm.run()

    print("\n========== FINAL RESULT ==========")

    print(f"Population Size : {len(population)}")

    best = population.best()

    print(f"Makespan : {best.makespan}")
    print(f"Cost      : {best.total_cost}")
    print(f"Carbon    : {best.total_carbon}")
    print(f"Energy    : {best.total_energy}")
    print(f"Rank      : {best.rank}")
    print(f"Crowding  : {best.crowding_distance}")


if __name__ == "__main__":
    main()