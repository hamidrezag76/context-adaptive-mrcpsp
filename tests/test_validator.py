from pathlib import Path

from src.parser.mm_parser import MMParser
from src.validation.project_validator import ProjectValidator


def main():

    parser = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    )

    project = parser.parse()
    print("Activities =", len(project.activities))

    for activity in project.activities.values():
        print("First activity:", activity.id)
        print("Modes:", len(activity.modes))
        break

    validator = ProjectValidator()

    result = validator.validate(project)

    print(result.valid)

    print(result.errors)

    print(result.warnings)


if __name__ == "__main__":

    main()
