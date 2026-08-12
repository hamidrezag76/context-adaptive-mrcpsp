from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def main():

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    algorithm = NSGA2(
        project=project,
        population_size=20,
        generations=20,
        seed=1,
        context_adaptive=True,
    )

    algorithm.run()

    print()
    print("========== CA HISTORY ==========")

    print(
        "History records:",
        len(algorithm.history),
    )

    print()

    for row in algorithm.history:

        print(
            f"Gen={row['generation']:2d} | "
            f"carbon={row['carbon_pressure']:.3f} | "
            f"energy={row['energy_pressure']:.3f} | "
            f"resource={row['resource_pressure']:.3f} | "
            f"cost={row['cost_pressure']:.3f} | "
            f"schedule={row['schedule_pressure']:.3f} | "
            f"uncertainty={row['uncertainty']:.3f} | "
            f"pc={row['crossover_probability']:.3f} | "
            f"pm={row['mutation_probability']:.3f} | "
            f"makespan={row['best_makespan']:.1f}"
        )


if __name__ == "__main__":
    main()
