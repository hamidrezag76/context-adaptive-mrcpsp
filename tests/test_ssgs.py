from pathlib import Path

from src.parser.mm_parser import MMParser
from src.scheduling.ssgs import SSGS


def main():

    parser = MMParser(

        Path("benchmarks/data/j3010_1.mm")

    )

    project = parser.parse()

    priority = [

        a.id

        for a in project.activities.values()

    ]
    mode_assignment = {}

    for a in project.activities.values():

        mode_assignment[a.id] = a.modes[0].id

    ssgs = SSGS(project)

    schedule = ssgs.generate(

        priority,

        mode_assignment

    )

    for s in schedule[:10]:

        print(s)


if __name__ == "__main__":

    main()
