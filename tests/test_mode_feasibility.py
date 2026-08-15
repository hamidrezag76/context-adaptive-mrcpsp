from src.parser.mm_parser import MMParser
from src.optimization.population_initializer import PopulationInitializer


def is_mode_feasible(project, activity_id, mode_id):

    activity = project.get_activity(activity_id)

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


def test_initializer_never_selects_infeasible_mode():

    project = MMParser(
        "benchmarks/data/j3033_9.mm"
    ).parse()

    initializer = PopulationInitializer(
        project,
        seed=42,
    )

    population = initializer.initialize(
        population_size=100,
    )

    for chromosome in population:

        for activity_id, mode_id in (
            chromosome.mode_assignment.items()
        ):

            assert is_mode_feasible(
                project,
                activity_id,
                mode_id,
            ), (
                f"Infeasible mode selected: "
                f"activity={activity_id}, "
                f"mode={mode_id}"
            )


if __name__ == "__main__":

    test_initializer_never_selects_infeasible_mode()

    print(
        "INITIALIZER MODE FEASIBILITY TEST: PASS"
    )
