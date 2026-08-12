from types import SimpleNamespace

from src.context.context_manager import ContextManager
from src.models.project import Project


def make_chromosome(
    makespan,
    cost,
    carbon,
    energy,
):
    return SimpleNamespace(
        makespan=float(makespan),
        total_cost=float(cost),
        total_carbon=float(carbon),
        total_energy=float(energy),
        feasible=True,
        decoded_schedule=None,
    )


def make_population(chromosomes):
    return SimpleNamespace(
        individuals=chromosomes
    )


def test_context_changes_when_population_changes():

    # --------------------------------------------------
    # Reference project
    # --------------------------------------------------

    project = Project(
        horizon=100,
        reference_cost=1000.0,
        reference_carbon=1000.0,
        reference_energy=1000.0,
        baseline_cost=1000.0,
        baseline_carbon=1000.0,
        baseline_energy=1000.0,
    )

    manager = ContextManager(
        project=project,
        seed=42,
    )

    # --------------------------------------------------
    # Generation 0
    # --------------------------------------------------

    population_0 = make_population(
        [
            make_chromosome(
                makespan=20,
                cost=100,
                carbon=100,
                energy=100,
            ),
            make_chromosome(
                makespan=20,
                cost=100,
                carbon=100,
                energy=100,
            ),
        ]
    )

    context_0 = manager.update(
        population=population_0,
        generation=0,
        max_generations=10,
    )

    # --------------------------------------------------
    # Generation 1
    # --------------------------------------------------

    population_1 = make_population(
        [
            make_chromosome(
                makespan=40,
                cost=200,
                carbon=200,
                energy=200,
            ),
            make_chromosome(
                makespan=40,
                cost=200,
                carbon=200,
                energy=200,
            ),
        ]
    )

    context_1 = manager.update(
        population=population_1,
        generation=1,
        max_generations=10,
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print()
    print("========== CONTEXT DYNAMICS TEST ==========")

    print(
        "Generation 0:",
        context_0,
    )

    print(
        "Generation 1:",
        context_1,
    )

    print(
        "Context 0:",
        context_0.as_vector(),
    )

    print(
        "Context 1:",
        context_1.as_vector(),
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert (
        context_0.as_vector()
        != context_1.as_vector()
    )

    assert (
        context_1.cost_pressure
        > context_0.cost_pressure
    )

    assert (
        context_1.carbon_pressure
        > context_0.carbon_pressure
    )

    assert (
        context_1.energy_pressure
        > context_0.energy_pressure
    )

    assert (
        context_1.schedule_pressure
        > context_0.schedule_pressure
    )