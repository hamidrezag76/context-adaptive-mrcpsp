from pathlib import Path

from src.parser.mm_parser import MMParser
from src.scheduling.network_analysis import NetworkAnalysis


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    analysis = NetworkAnalysis(project)

    analysis.analyze()

    print(analysis.critical)


if __name__ == "__main__":
    main()