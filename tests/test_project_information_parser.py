from pathlib import Path

from src.models.project import Project
from src.parser.psplib_reader import PSPLIBReader
from src.parser.project_information_parser import ProjectInformationParser


reader = PSPLIBReader(
    Path("benchmarks/data/j301_1.mm")
)

reader.read()

project = Project()

ProjectInformationParser(
    reader
).parse(
    project
)

print(project.horizon)