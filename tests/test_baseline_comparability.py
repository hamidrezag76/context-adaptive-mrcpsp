"""
Baseline vs Context-Adaptive NSGA-II comparability test.

The two algorithms must use identical:
    - benchmark
    - population size
    - number of generations
    - random seed
    - initial population

They must differ only through context adaptation.
"""

from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def test_baseline_comparability():

    # --------------------------------------------------
    # Experimental configuration
    # --------------------------------------------------

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    population_size = 20
    generations = 20
    seed = 42

    # --------------------------------------------------
    # Create two algorithms
    # --------------------------------------------------

    baseline = NSGA2(
        project=project,
        population_size=population_size,
        generations=generations,
        seed=seed,
        context_adaptive=False,
    )

    adaptive = NSGA2(
        project=project,
        population_size=population_size,
        generations=generations,
        seed=seed,
        context_adaptive=True,
    )

    # --------------------------------------------------
    # Configuration comparability
    # --------------------------------------------------

    assert baseline.population_size == adaptive.population_size
    assert baseline.generations == adaptive.generations
    assert baseline.seed == adaptive.seed

    assert baseline.context_adaptive is False
    assert adaptive.context_adaptive is True

    # --------------------------------------------------
    # Initialization comparability
    # --------------------------------------------------

    baseline.initialize()
    adaptive.initialize()

    baseline_initial = [
        (
            chromosome.priority_list.copy(),
            chromosome.mode_assignment.copy(),
        )
        for chromosome in baseline.population.individuals
    ]

    adaptive_initial = [
        (
            chromosome.priority_list.copy(),
            chromosome.mode_assignment.copy(),
        )
        for chromosome in adaptive.population.individuals
    ]

    assert baseline_initial == adaptive_initial

    # --------------------------------------------------
    # Run baseline
    # --------------------------------------------------

    baseline = NSGA2(
        project=project,
        population_size=population_size,
        generations=generations,
        seed=seed,
        context_adaptive=False,
    )

    baseline.run()

    # --------------------------------------------------
    # Run adaptive method
    # --------------------------------------------------

    adaptive = NSGA2(
        project=project,
        population_size=population_size,
        generations=generations,
        seed=seed,
        context_adaptive=True,
    )

    adaptive.run()

    # --------------------------------------------------
    # Population-size consistency
    # --------------------------------------------------

    assert len(
        baseline.population.individuals
    ) == population_size

    assert len(
        adaptive.population.individuals
    ) == population_size

    # --------------------------------------------------
    # History length
    # --------------------------------------------------

    assert len(
        baseline.history
    ) == generations + 1

    assert len(
        adaptive.history
    ) == generations + 1

    # --------------------------------------------------
    # Baseline operator parameters must remain fixed
    # --------------------------------------------------

    baseline_pc = [
        row["crossover_probability"]
        for row in baseline.history
    ]

    baseline_pm = [
        row["mutation_probability"]
        for row in baseline.history
    ]

    assert all(
        pc == 0.90
        for pc in baseline_pc
    )

    assert all(
        pm == 0.15
        for pm in baseline_pm
    )

    # --------------------------------------------------
    # Adaptive operator parameters
    # --------------------------------------------------

    adaptive_pc = [
        row["crossover_probability"]
        for row in adaptive.history
    ]

    adaptive_pm = [
        row["mutation_probability"]
        for row in adaptive.history
    ]

    # Bounds

    assert all(
        0.50 <= pc <= 0.95
        for pc in adaptive_pc
    )

    assert all(
        0.05 <= pm <= 0.60
        for pm in adaptive_pm
    )

    # --------------------------------------------------
    # CA adaptation must actually differ from baseline
    # --------------------------------------------------

    assert any(
        abs(pc - 0.90) > 1e-12
        for pc in adaptive_pc
    )

    assert any(
        abs(pm - 0.15) > 1e-12
        for pm in adaptive_pm
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE COMPARABILITY")
    print("=" * 70)

    print(
        "Initial populations identical:",
        baseline_initial == adaptive_initial,
    )

    print(
        "Baseline Pc:",
        baseline_pc,
    )

    print(
        "Baseline Pm:",
        baseline_pm,
    )

    print(
        "Adaptive Pc:",
        [
            round(x, 6)
            for x in adaptive_pc
        ],
    )

    print(
        "Adaptive Pm:",
        [
            round(x, 6)
            for x in adaptive_pm
        ],
    )

    print()
    print("BASELINE COMPARABILITY: PASS")