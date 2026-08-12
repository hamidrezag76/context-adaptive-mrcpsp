from pathlib import Path

from src.models.project import Project
from src.parser.psplib_reader import PSPLIBReader
from src.parser.precedence_parser import PrecedenceParser
from src.parser.mm_parser import MMParser


reader = PSPLIBReader(
    Path("benchmarks/data/j301_1.mm")
)

reader.read()

project = MMParser(
    Path("benchmarks/data/j301_1.mm")
).parse()

PrecedenceParser(
    reader
).parse(
    project
)

print(project.total_edges)

a = project.get_activity(2)

print(a.successors)

b = project.get_activity(5)

print(b.predecessors)