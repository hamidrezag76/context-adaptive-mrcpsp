from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.chromosome import Chromosome
from src.optimization.operators.mutation import Mutation
from src.optimization.operators.repair import Repair

def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    priority = list(project.activities.keys())

    mode_assignment = {
        a.id: a.modes[0].id
        for a in project.activities.values()
    }

    chromosome = Chromosome(
        priority_list=priority,
        mode_assignment=mode_assignment,
    )
    mutation = Mutation(project)

    repair = Repair(project)

    child = mutation.apply(chromosome)

    repaired = repair.apply(child)

    print("Activities:", len(repaired.priority_list))
    print("Unique:", len(set(repaired.priority_list)))
    print("Modes:", len(repaired.mode_assignment))

    assert len(repaired.priority_list) == len(project.activities)
    assert len(set(repaired.priority_list)) == len(project.activities)
    assert len(repaired.mode_assignment) == len(project.activities)

    print("Repair OK")


if __name__ == "__main__":
    main()