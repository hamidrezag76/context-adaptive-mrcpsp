from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def run(seed):

    project = MMParser(
        PROJECT_ROOT / "benchmarks" / "data" / "j3010_1.mm"
    ).parse()

    algorithm = NSGA2(
        project,
        population_size=30,
        generations=20,
        seed=seed,
        context_adaptive=True,
    )

    population = algorithm.run()

    return sorted(
        (
            round(c.makespan, 6),
            round(c.total_cost, 6),
            round(c.total_carbon, 6),
            round(c.total_energy, 6),
        )
        for c in population.individuals
    )


def main():

    print("=" * 70)
    print("NSGA-II REPRODUCIBILITY VALIDATION")
    print("=" * 70)

    result_1 = run(42)
    result_2 = run(42)
    result_3 = run(123)

    print()
    print("Same seed identical:", result_1 == result_2)
    print("Different seed different:", result_1 != result_3)

    if result_1 != result_2:
        raise AssertionError(
            "Same seed produced different results."
        )

    if result_1 == result_3:
        raise AssertionError(
            "Different seeds produced identical results."
        )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
