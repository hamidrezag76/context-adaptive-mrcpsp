from pathlib import Path

from src.parser.mm_parser import MMParser


def test_parser():

    file = Path("benchmarks/data/j3010_1.mm")

    parser = MMParser(file)

    project = parser.parse()

    print("Activities:", len(project.activities))

    first_activity = project.activities_list[0]

    print(first_activity)

    assert len(project.activities) == 32
    assert first_activity.id == 1