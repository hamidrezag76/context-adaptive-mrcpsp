from src.context.context import Context
from src.context.operator_controller import OperatorController


def test_operator_adaptation():

    controller = OperatorController()

    # --------------------------------------------------
    # Low-pressure context
    # --------------------------------------------------

    low = Context(
        carbon_pressure=0.10,
        energy_pressure=0.10,
        resource_pressure=0.10,
        cost_pressure=0.10,
        schedule_pressure=0.10,
        uncertainty=0.10,
    )

    # --------------------------------------------------
    # High-pressure context
    # --------------------------------------------------

    high = Context(
        carbon_pressure=0.90,
        energy_pressure=0.90,
        resource_pressure=0.90,
        cost_pressure=0.90,
        schedule_pressure=0.90,
        uncertainty=0.90,
    )

    # --------------------------------------------------
    # Compute adaptive probabilities
    # --------------------------------------------------

    low_pc = controller.crossover_probability(low)
    low_pm = controller.mutation_probability(low)

    high_pc = controller.crossover_probability(high)
    high_pm = controller.mutation_probability(high)

    print()
    print("========== OPERATOR ADAPTATION ==========")

    print("Low-pressure context:")
    print("  Crossover :", low_pc)
    print("  Mutation  :", low_pm)

    print()

    print("High-pressure context:")
    print("  Crossover :", high_pc)
    print("  Mutation  :", high_pm)

    # --------------------------------------------------
    # Expected adaptation
    # --------------------------------------------------

    assert high_pm > low_pm

    assert high_pc < low_pc

    # --------------------------------------------------
    # Probability bounds
    # --------------------------------------------------

    assert 0.50 <= low_pc <= 0.95
    assert 0.50 <= high_pc <= 0.95

    assert 0.05 <= low_pm <= 0.60
    assert 0.05 <= high_pm <= 0.60