from src.context.context import Context
from src.context.operator_controller import OperatorController


def test_neutral_context_returns_base_probabilities():

    controller = OperatorController()

    context = Context(
        carbon_pressure=0.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    assert (
        controller.crossover_probability(context)
        == 0.90
    )

    assert (
        controller.mutation_probability(context)
        == 0.15
    )


def test_uncertainty_reduces_crossover_and_increases_mutation():

    controller = OperatorController()

    low = Context(
        carbon_pressure=0.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    high = Context(
        carbon_pressure=0.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=1.0,
    )

    low_pc = controller.crossover_probability(low)
    high_pc = controller.crossover_probability(high)

    low_pm = controller.mutation_probability(low)
    high_pm = controller.mutation_probability(high)

    assert high_pc < low_pc
    assert high_pm > low_pm


def test_each_pressure_reduces_crossover():

    controller = OperatorController()

    neutral = Context(
        carbon_pressure=0.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    base_pc = controller.crossover_probability(
        neutral
    )

    pressure_names = (
        "carbon_pressure",
        "energy_pressure",
        "resource_pressure",
        "cost_pressure",
        "schedule_pressure",
        "uncertainty",
    )

    for name in pressure_names:

        values = {
            "carbon_pressure": 0.0,
            "energy_pressure": 0.0,
            "resource_pressure": 0.0,
            "cost_pressure": 0.0,
            "schedule_pressure": 0.0,
            "uncertainty": 0.0,
        }

        values[name] = 1.0

        context = Context(**values)

        pc = controller.crossover_probability(
            context
        )

        assert pc < base_pc, (
            f"{name} failed to reduce crossover."
        )


def test_each_pressure_increases_mutation():

    controller = OperatorController()

    neutral = Context(
        carbon_pressure=0.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    base_pm = controller.mutation_probability(
        neutral
    )

    pressure_names = (
        "carbon_pressure",
        "energy_pressure",
        "resource_pressure",
        "cost_pressure",
        "schedule_pressure",
        "uncertainty",
    )

    for name in pressure_names:

        values = {
            "carbon_pressure": 0.0,
            "energy_pressure": 0.0,
            "resource_pressure": 0.0,
            "cost_pressure": 0.0,
            "schedule_pressure": 0.0,
            "uncertainty": 0.0,
        }

        values[name] = 1.0

        context = Context(**values)

        pm = controller.mutation_probability(
            context
        )

        assert pm > base_pm, (
            f"{name} failed to increase mutation."
        )


def test_operator_probability_bounds():

    controller = OperatorController()

    maximum = Context(
        carbon_pressure=1.0,
        energy_pressure=1.0,
        resource_pressure=1.0,
        cost_pressure=1.0,
        schedule_pressure=1.0,
        uncertainty=1.0,
    )

    pc = controller.crossover_probability(
        maximum
    )

    pm = controller.mutation_probability(
        maximum
    )

    assert 0.50 <= pc <= 0.95
    assert 0.05 <= pm <= 0.60

    assert abs(pc - 0.54) < 1e-12
    assert abs(pm - 0.60) < 1e-12


def test_operator_probabilities_match_full_formula():

    controller = OperatorController()

    context = Context(
        carbon_pressure=0.4,
        energy_pressure=0.3,
        resource_pressure=0.8,
        cost_pressure=0.5,
        schedule_pressure=0.2,
        uncertainty=0.7,
    )

    expected_pc = (
        0.90
        - 0.12 * 0.7
        - 0.08 * 0.8
        - 0.05 * 0.2
        - 0.04 * 0.5
        - 0.04 * 0.4
        - 0.03 * 0.3
    )

    expected_pm = (
        0.15
        + 0.20 * 0.7
        + 0.12 * 0.8
        + 0.10 * 0.2
        + 0.08 * 0.4
        + 0.06 * 0.3
        + 0.05 * 0.5
    )

    assert (
        abs(
            controller.crossover_probability(context)
            - expected_pc
        )
        < 1e-12
    )

    assert (
        abs(
            controller.mutation_probability(context)
            - expected_pm
        )
        < 1e-12
    )
