from pathlib import Path

from src.parser.mm_parser import MMParser
from src.scheduling.network_analysis import NetworkAnalysis


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    analysis = NetworkAnalysis(project)

    analysis.analyze()

    for activity in project.ordered_activities[:10]:

        print(

            activity.id,

            activity.earliest_start,

            activity.latest_start,

        )


if __name__ == "__main__":

    main()