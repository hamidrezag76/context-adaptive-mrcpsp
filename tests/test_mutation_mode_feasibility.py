from src.parser.mm_parser import MMParser
from src.optimization.chromosome import Chromosome
from src.optimization.operators.mutation import Mutation


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


def test_mutation_never_creates_infeasible_mode():

    project = MMParser(
        "benchmarks/data/j3033_9.mm"
    ).parse()

    mutation = Mutation(
        project,
        probability=1.0,
        seed=42,
    )

    chromosome = Chromosome(
        priority_list=list(
            project.activities.keys()
        ),
        mode_assignment={
            activity.id: next(
                mode.id
                for mode in activity.modes
                if is_mode_feasible(
                    project,
                    activity.id,
                    mode.id,
                )
            )
            for activity
            in project.activities.values()
        },
    )

    for _ in range(1000):

        child = mutation.apply(
            chromosome
        )

        for activity_id, mode_id in (
            child.mode_assignment.items()
        ):

            assert is_mode_feasible(
                project,
                activity_id,
                mode_id,
            ), (
                f"Infeasible mode created: "
                f"activity={activity_id}, "
                f"mode={mode_id}"
            )


if __name__ == "__main__":

    test_mutation_never_creates_infeasible_mode()

    print(
        "MUTATION MODE FEASIBILITY TEST: PASS"
    )
