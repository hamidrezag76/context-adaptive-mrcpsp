from pathlib import Path
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2
from src.validation.schedule_validator import ScheduleValidator


def main():

    print("=" * 70)
    print("SCHEDULE FEASIBILITY VALIDATION")
    print("=" * 70)

    project = MMParser(
        PROJECT_ROOT / "benchmarks" / "data" / "j3010_1.mm"
    ).parse()

    algorithm = NSGA2(
        project,
        population_size=30,
        generations=20,
        seed=42,
        context_adaptive=True,
    )

    population = algorithm.run()

    validator = ScheduleValidator(project)

    checked = 0

    for chromosome in population.individuals:

        decoded = chromosome.decoded_schedule

        if decoded is None:
            raise ValueError(
                "Chromosome has no decoded schedule."
            )

        validator.validate(decoded)

        checked += 1

    print()
    print(f"Population checked: {checked}")
    print("Precedence: PASS")
    print("Timing: PASS")
    print("Renewable resources: PASS")
    print("Mode assignments: PASS")
    print("Makespan consistency: PASS")
    print()
    print("ALL SCHEDULES ARE FEASIBLE.")


if __name__ == "__main__":
    main()
