from pathlib import Path

from src.parser.mm_parser import MMParser
from src.scheduling.ssgs import SSGS


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    mode_assignment = {}

    for activity in project.activities.values():

        mode_assignment[activity.id] = activity.modes[0].id

    priority = project.topological_sort()

    result = SSGS(project).generate(

        priority,

        mode_assignment,

    )

    print(type(result))

    print(result.makespan)

    print(result.feasible)

    print(len(result.schedule))


if __name__ == "__main__":
    main()