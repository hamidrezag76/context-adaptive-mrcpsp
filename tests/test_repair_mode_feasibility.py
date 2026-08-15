from src.parser.mm_parser import MMParser
from src.optimization.chromosome import Chromosome
from src.optimization.operators.repair import Repair


def is_mode_feasible(
    project,
    activity_id,
    mode_id,
):

    activity = project.get_activity(
        activity_id
    )

    mode = next(
        mode
        for mode in activity.modes
        if mode.id == mode_id
    )

    return all(
        requirement <= capacity
        for requirement, capacity
        in zip(
            mode.renewable,
            project.renewable_capacities,
        )
    )


def test_repair_removes_infeasible_modes():

    project = MMParser(
        "benchmarks/data/j3033_9.mm"
    ).parse()

    chromosome = Chromosome(
        priority_list=list(
            project.activities.keys()
        ),
        mode_assignment={
            activity.id: activity.modes[0].id
            for activity
            in project.activities.values()
        },
    )

    repair = Repair(
        project
    )

    repaired = repair.apply(
        chromosome
    )

    for activity_id, mode_id in (
        repaired.mode_assignment.items()
    ):

        assert is_mode_feasible(
            project,
            activity_id,
            mode_id,
        ), (
            f"Repair left infeasible mode: "
            f"activity={activity_id}, "
            f"mode={mode_id}"
        )


if __name__ == "__main__":

    test_repair_removes_infeasible_modes()

    print(
        "REPAIR MODE FEASIBILITY TEST: PASS"
    )
