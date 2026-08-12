from pathlib import Path

from src.parser.mm_parser import MMParser


def main():

    parser = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    )

    project = parser.parse()

    for activity in project.activities.values():

        print(f"\nActivity {activity.id}")

        for mode in activity.modes:

            print(
                mode.id,
                mode.duration,
                mode.renewable,
                mode.nonrenewable,
            )


if __name__ == "__main__":
    main()
