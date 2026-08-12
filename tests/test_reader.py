from pathlib import Path

from src.parser.psplib_reader import PSPLIBReader


reader = PSPLIBReader(
    Path("benchmarks/data/j301_1.mm")
)

reader.read()

print(reader.find_section("PROJECT INFORMATION"))
print(reader.find_section("PRECEDENCE RELATIONS"))
print(reader.find_section("REQUESTS/DURATIONS"))
print(reader.find_section("RESOURCEAVAILABILITIES"))