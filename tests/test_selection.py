from src.context.context import Context
from src.optimization.chromosome import Chromosome
from src.optimization.operators.selection import TournamentSelection


def make_chromosome(
    *,
    makespan: float = 10.0,
    cost: float = 100.0,
    carbon: float = 100.0,
    energy: float = 100.0,
    rank: int = 0,
    crowding: float = 0.0,
) -> Chromosome:
    """
    Create a chromosome for deterministic selection tests.
    """

    chromosome = Chromosome(
        priority_list=[1, 2, 3],
        mode_assignment={1: 1},
    )

    chromosome.makespan = makespan
    chromosome.total_cost = cost
    chromosome.total_carbon = carbon
    chromosome.total_energy = energy

    chromosome.rank = rank
    chromosome.crowding_distance = crowding

    return chromosome


def test_standard_selection_prioritizes_lower_rank():
    """
    Standard NSGA-II selection must prioritize lower Pareto rank.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=False,
    )

    better_rank = make_chromosome(
        rank=0,
        crowding=0.0,
    )

    worse_rank = make_chromosome(
        rank=1,
        crowding=100.0,
    )

    population = [
        better_rank,
        worse_rank,
    ]

    selected = selector._better(
        better_rank,
        worse_rank,
        population=population,
        context=None,
    )

    assert selected is better_rank


def test_standard_selection_uses_crowding_when_rank_is_equal():
    """
    Standard NSGA-II selection must use larger crowding distance
    when Pareto ranks are equal.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=False,
    )

    low_crowding = make_chromosome(
        rank=0,
        crowding=1.0,
    )

    high_crowding = make_chromosome(
        rank=0,
        crowding=5.0,
    )

    population = [
        low_crowding,
        high_crowding,
    ]

    selected = selector._better(
        low_crowding,
        high_crowding,
        population=population,
        context=None,
    )

    assert selected is high_crowding


def test_context_selection_prefers_objective_under_high_pressure():
    """
    Context-aware selection must prefer the chromosome with the
    better objective corresponding to the dominant context pressure.

    Here carbon pressure is maximal while all other pressures are zero.
    Therefore lower carbon should be preferred.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=True,
    )

    low_carbon = make_chromosome(
        carbon=100.0,
        rank=0,
        crowding=0.0,
    )

    high_carbon = make_chromosome(
        carbon=200.0,
        rank=0,
        crowding=0.0,
    )

    population = [
        low_carbon,
        high_carbon,
    ]

    context = Context(
        carbon_pressure=1.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    selected = selector._better(
        low_carbon,
        high_carbon,
        population=population,
        context=context,
    )

    assert selected is low_carbon


def test_context_selection_does_not_override_pareto_rank():
    """
    Context preference must never override the primary NSGA-II
    Pareto-rank criterion.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=True,
    )

    better_rank = make_chromosome(
        carbon=200.0,
        rank=0,
        crowding=0.0,
    )

    worse_rank = make_chromosome(
        carbon=100.0,
        rank=1,
        crowding=100.0,
    )

    population = [
        better_rank,
        worse_rank,
    ]

    context = Context(
        carbon_pressure=1.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    selected = selector._better(
        better_rank,
        worse_rank,
        population=population,
        context=context,
    )

    assert selected is better_rank


def test_context_preference_is_scale_normalized():
    """
    Context preference must normalize objectives before weighting
    them, preventing cost/carbon/energy magnitudes from dominating
    makespan merely because of their numerical scale.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=True,
    )

    a = make_chromosome(
        makespan=10.0,
        cost=100.0,
        carbon=100.0,
        energy=100.0,
    )

    b = make_chromosome(
        makespan=20.0,
        cost=200.0,
        carbon=200.0,
        energy=200.0,
    )

    population = [a, b]

    context = Context(
        carbon_pressure=1.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    score_a = selector._context_preference(
        a,
        population,
        context,
    )

    score_b = selector._context_preference(
        b,
        population,
        context,
    )

    assert score_a < score_b


def test_zero_context_pressure_falls_back_to_equal_weights():
    """
    When all four objective-related context pressures are zero,
    the preference calculation must fall back to equal weights.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=True,
    )

    a = make_chromosome(
        makespan=10.0,
        cost=100.0,
        carbon=100.0,
        energy=100.0,
    )

    b = make_chromosome(
        makespan=20.0,
        cost=200.0,
        carbon=200.0,
        energy=200.0,
    )

    population = [a, b]

    context = Context(
        carbon_pressure=0.0,
        energy_pressure=0.0,
        resource_pressure=0.0,
        cost_pressure=0.0,
        schedule_pressure=0.0,
        uncertainty=0.0,
    )

    score_a = selector._context_preference(
        a,
        population,
        context,
    )

    score_b = selector._context_preference(
        b,
        population,
        context,
    )

    assert score_a < score_b


def test_select_pair_returns_distinct_parents():
    """
    Parent selection must return two distinct chromosome objects.
    """

    selector = TournamentSelection(
        seed=42,
        context_adaptive=False,
    )

    population = [
        make_chromosome(
            rank=0,
            crowding=float(i),
        )
        for i in range(10)
    ]

    parent1, parent2 = selector.select_pair(
        population,
    )

    assert parent1 is not parent2