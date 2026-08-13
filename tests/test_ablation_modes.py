from pathlib import Path

from src.models.project import Project
from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


INSTANCE = Path("benchmarks/data/j3010_1.mm")


def run_mode(
    *,
    context_adaptive: bool,
    operator_adaptive: bool,
):
    project = MMParser(INSTANCE).parse()

    algorithm = NSGA2(
        project=project,
        population_size=10,
        generations=5,
        seed=42,
        context_adaptive=context_adaptive,
        operator_adaptive=operator_adaptive,
    )

    population = algorithm.run()

    return algorithm, population


def extract_operator_history(algorithm):
    crossover = [
        float(row["crossover_probability"])
        for row in algorithm.history
    ]

    mutation = [
        float(row["mutation_probability"])
        for row in algorithm.history
    ]

    return crossover, mutation


def test_ablation_modes():

    # =========================================================
    # BASELINE
    # =========================================================

    baseline, baseline_population = run_mode(
        context_adaptive=False,
        operator_adaptive=False,
    )

    baseline_pc, baseline_pm = extract_operator_history(
        baseline
    )

    assert baseline_population

    assert baseline_pc
    assert baseline_pm

    assert all(
        abs(value - 0.90) < 1e-12
        for value in baseline_pc
    )

    assert all(
        abs(value - 0.15) < 1e-12
        for value in baseline_pm
    )

    # =========================================================
    # CONTEXT-ONLY
    # =========================================================

    context_only, context_population = run_mode(
        context_adaptive=True,
        operator_adaptive=False,
    )

    context_pc, context_pm = extract_operator_history(
        context_only
    )

    assert context_population

    assert context_pc
    assert context_pm

    # Operators must remain fixed.
    assert all(
        abs(value - 0.90) < 1e-12
        for value in context_pc
    )

    assert all(
        abs(value - 0.15) < 1e-12
        for value in context_pm
    )

    # Context must actually be recorded.
    assert context_only.history

    # =========================================================
    # FULL CA
    # =========================================================

    full_ca, full_population = run_mode(
        context_adaptive=True,
        operator_adaptive=True,
    )

    full_pc, full_pm = extract_operator_history(
        full_ca
    )

    assert full_population

    assert full_pc
    assert full_pm

    # Adaptive operators must stay within controller bounds.
    assert all(
        0.50 <= value <= 0.95
        for value in full_pc
    )

    assert all(
        0.05 <= value <= 0.60
        for value in full_pm
    )

    # Full CA must actually adapt at least one operator.
    assert any(
        abs(value - 0.90) > 1e-12
        for value in full_pc
    )

    assert any(
        abs(value - 0.15) > 1e-12
        for value in full_pm
    )

    print()
    print("=" * 70)
    print("ABLATION MODE VALIDATION")
    print("=" * 70)

    print(
        "Baseline Pc:",
        baseline_pc,
    )

    print(
        "Baseline Pm:",
        baseline_pm,
    )

    print(
        "Context-only Pc:",
        context_pc,
    )

    print(
        "Context-only Pm:",
        context_pm,
    )

    print(
        "Full CA Pc:",
        full_pc,
    )

    print(
        "Full CA Pm:",
        full_pm,
    )

    print()
    print("Baseline mode: PASS")
    print("Context-only mode: PASS")
    print("Full CA mode: PASS")
    print("Ablation mode validation: PASS")