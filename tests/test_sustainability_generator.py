from pathlib import Path

from src.parser.mm_parser import MMParser
from src.context.sustainability_generator import SustainabilityGenerator


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    print(project.reference_cost)
    print(project.reference_carbon)
    print(project.reference_energy)

    for activity in project.activities.values():

        if activity.id == 1:
            continue

        print(f"\nActivity {activity.id}")

        for mode in activity.modes:

            print(
                mode.id,
                mode.duration,
                mode.cost,
                mode.carbon,
                mode.energy,
            )

        break

if __name__ == "__main__":
    main()