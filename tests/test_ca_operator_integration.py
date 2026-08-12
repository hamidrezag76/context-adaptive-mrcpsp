from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def test_ca_operator_integration():

    # --------------------------------------------------
    # Load benchmark
    # --------------------------------------------------

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    # --------------------------------------------------
    # Run CA-NSGA-II
    # --------------------------------------------------

    algorithm = NSGA2(
        project=project,
        population_size=20,
        generations=5,
        seed=42,
        context_adaptive=True,
    )

    pareto = algorithm.run()

    # --------------------------------------------------
    # Basic result validation
    # --------------------------------------------------

    assert pareto

    # --------------------------------------------------
    # Extract operator history
    # --------------------------------------------------

    history = getattr(
        algorithm,
        "history",
        None,
    )

    print()
    print("=" * 70)
    print("CA-NSGA-II OPERATOR ADAPTATION")
    print("=" * 70)

    print("Pareto solutions:", len(pareto))
    print("History type:", type(history).__name__)

    # --------------------------------------------------
    # Validate history availability
    # --------------------------------------------------

    assert history is not None

    print("History entries:", len(history))

    # --------------------------------------------------
    # Inspect history
    # --------------------------------------------------

    for i, row in enumerate(history):

        print(
            f"Generation {i}:",
            row,
        )

    # --------------------------------------------------
    # Extract crossover / mutation probabilities
    # --------------------------------------------------

    crossover_values = []
    mutation_values = []

    for row in history:

        if isinstance(row, dict):

            if "crossover_probability" in row:

                crossover_values.append(
                    row["crossover_probability"]
                )

            if "mutation_probability" in row:

                mutation_values.append(
                    row["mutation_probability"]
                )

    print()
    print("Crossover history:", crossover_values)
    print("Mutation history :", mutation_values)

    # --------------------------------------------------
    # Validate adaptive parameters
    # --------------------------------------------------

    assert crossover_values
    assert mutation_values

    # Bounds

    assert all(
        0.50 <= x <= 0.95
        for x in crossover_values
    )

    assert all(
        0.05 <= x <= 0.60
        for x in mutation_values
    )