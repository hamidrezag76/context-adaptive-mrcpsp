from src.context.context import Context
from src.context.objective_weighting import ObjectiveWeighting


def test_objective_weights_sum_to_one():

    context = Context(
        carbon_pressure=0.8,
        energy_pressure=0.3,
        resource_pressure=0.5,
        cost_pressure=0.2,
        schedule_pressure=0.7,
        uncertainty=0.4,
    )

    weighting = ObjectiveWeighting()

    weights = weighting.compute(context)

    total = (
        weights.makespan
        + weights.cost
        + weights.carbon
        + weights.energy
    )

    assert abs(total - 1.0) < 1e-12


def test_objective_weights_are_normalized():

    context = Context(
        carbon_pressure=0.8,
        energy_pressure=0.3,
        resource_pressure=0.5,
        cost_pressure=0.2,
        schedule_pressure=0.7,
        uncertainty=0.4,
    )

    weighting = ObjectiveWeighting()

    weights = weighting.compute(context)

    values = [
        weights.makespan,
        weights.cost,
        weights.carbon,
        weights.energy,
    ]

    assert all(
        0.0 <= value <= 1.0
        for value in values
    )


def test_high_pressure_objective_receives_higher_weight():

    context = Context(
        carbon_pressure=0.9,
        energy_pressure=0.2,
        resource_pressure=0.5,
        cost_pressure=0.2,
        schedule_pressure=0.3,
        uncertainty=0.1,
    )

    weighting = ObjectiveWeighting()

    weights = weighting.compute(context)

    assert weights.carbon > weights.energy
    assert weights.carbon > weights.cost
    assert weights.carbon > weights.makespan