from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


def dominates(a, b):
    """
    Minimization dominance.
    a dominates b if it is no worse in all
    objectives and strictly better in at least one.
    """

    return (
        all(x <= y for x, y in zip(a.objectives, b.objectives))
        and any(x < y for x, y in zip(a.objectives, b.objectives))
    )


def main():

    print("\n========== ARCHIVE VALIDATION ==========")

    project = MMParser(
        Path("benchmarks/data/j3010_1.mm")
    ).parse()

    algorithm = NSGA2(
        project,
        population_size=20,
        generations=20,
        seed=42,
        context_adaptive=True,
    )

    algorithm.run()

    archive = algorithm.archive.archive

    print("Population size:", len(algorithm.population.individuals))
    print("Archive size:", len(archive))

    # -------------------------------------------------
    # 1. Duplicate check
    # -------------------------------------------------

    objective_vectors = [
        tuple(round(x, 8) for x in c.objectives)
        for c in archive
    ]

    duplicates = (
        len(objective_vectors)
        != len(set(objective_vectors))
    )

    print("Duplicate objective vectors:", duplicates)

    # -------------------------------------------------
    # 2. Internal dominance check
    # -------------------------------------------------

    dominated_pairs = []

    for i, a in enumerate(archive):

        for j, b in enumerate(archive):

            if i == j:
                continue

            if dominates(b, a):

                dominated_pairs.append(
                    (i + 1, j + 1)
                )

    print(
        "Internally dominated solutions:",
        len(dominated_pairs)
    )

    if dominated_pairs:

        print(
            "Dominated pairs:",
            dominated_pairs[:10]
        )

    # -------------------------------------------------
    # 3. Archive independence
    # -------------------------------------------------

    archive_before = [
        tuple(c.objectives)
        for c in archive
    ]

    # Change NSGA-II population attributes.
    for chromosome in algorithm.population.individuals:

        chromosome.rank = 999

        chromosome.crowding_distance = -999

    archive_after = [
        tuple(c.objectives)
        for c in algorithm.archive.archive
    ]

    independent = (
        archive_before == archive_after
    )

    print(
        "Archive independent of population:",
        independent
    )

    # -------------------------------------------------
    # Final result
    # -------------------------------------------------

    passed = (
        not duplicates
        and len(dominated_pairs) == 0
        and independent
    )

    print("\nVALIDATION RESULT:", "PASS" if passed else "FAIL")

    if not passed:

        raise AssertionError(
            "Elite Archive validation failed."
        )


if __name__ == "__main__":
    main()
