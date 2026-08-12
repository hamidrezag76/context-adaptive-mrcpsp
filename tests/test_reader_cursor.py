from pathlib import Path

from src.parser.psplib_reader import PSPLIBReader

reader = PSPLIBReader(
    Path("benchmarks/data/j301_1.mm")
)

reader.read()

reader.goto("PROJECT INFORMATION")

for _ in range(5):

    print(reader.next_line())