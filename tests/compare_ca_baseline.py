from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


INSTANCE = (
    PROJECT_ROOT
    / "benchmarks"
    / "data"
    / "j3010_1.mm"
)


def summarize(population):

    return {
        "population": len(population.individuals),
        "makespan": min(
            c.makespan
            for c in population.individuals
        ),
        "cost": min(
            c.total_cost
            for c in population.individuals
        ),
        "carbon": min(
            c.total_carbon
            for c in population.individuals
        ),
        "energy": min(
            c.total_energy
            for c in population.individuals
        ),
    }


def run(context_adaptive):

    project = MMParser(INSTANCE).parse()

    algorithm = NSGA2(
        project,
        population_size=30,
        generations=20,
        seed=42,
        context_adaptive=context_adaptive,
    )

    population = algorithm.run()

    return summarize(population)


def main():

    print("=" * 70)
    print("CA-NSGA-II vs BASELINE NSGA-II")
    print("=" * 70)

    ca = run(True)
    baseline = run(False)

    print()
    print("CA-NSGA-II")
    print("-" * 30)

    for key, value in ca.items():
        print(f"{key}: {value}")

    print()
    print("BASELINE NSGA-II")
    print("-" * 30)

    for key, value in baseline.items():
        print(f"{key}: {value}")

    print()
    print("DIFFERENCE")
    print("-" * 30)

    print(
        "Makespan:",
        ca["makespan"] - baseline["makespan"],
    )

    print(
        "Cost:",
        ca["cost"] - baseline["cost"],
    )

    print(
        "Carbon:",
        ca["carbon"] - baseline["carbon"],
    )

    print(
        "Energy:",
        ca["energy"] - baseline["energy"],
    )


if __name__ == "__main__":
    main()
