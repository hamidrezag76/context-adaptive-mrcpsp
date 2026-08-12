from src.experiments.objective_normalizer import (
    ObjectiveNormalizer,
)


def main():

    print(
        "\n========== OBJECTIVE NORMALIZATION TEST =========="
    )

    baseline = [
        (10.0, 100.0, 50.0, 80.0),
        (8.0, 120.0, 40.0, 70.0),
    ]

    adaptive = [
        (9.0, 110.0, 45.0, 75.0),
        (7.0, 130.0, 35.0, 65.0),
    ]

    normalizer = ObjectiveNormalizer.from_sets(
        baseline,
        adaptive,
    )

    print(
        "Minimum:",
        normalizer.bounds.minimum,
    )

    print(
        "Maximum:",
        normalizer.bounds.maximum,
    )

    baseline_normalized = (
        normalizer.normalize_set(
            baseline
        )
    )

    adaptive_normalized = (
        normalizer.normalize_set(
            adaptive
        )
    )

    print(
        "Baseline normalized:",
        baseline_normalized,
    )

    print(
        "Adaptive normalized:",
        adaptive_normalized,
    )

    assert normalizer.bounds.minimum == (
        7.0,
        100.0,
        35.0,
        65.0,
    )

    assert normalizer.bounds.maximum == (
        10.0,
        130.0,
        50.0,
        80.0,
    )

    all_points = (
        baseline_normalized
        + adaptive_normalized
    )

    assert all(
        0.0 <= value <= 1.0
        for point in all_points
        for value in point
    )

    assert (
        normalizer.normalize(
            normalizer.bounds.minimum
        )
        == (0.0, 0.0, 0.0, 0.0)
    )

    assert (
        normalizer.normalize(
            normalizer.bounds.maximum
        )
        == (1.0, 1.0, 1.0, 1.0)
    )

    print(
        "All normalized values in [0,1]: PASS"
    )

    print(
        "Common normalization: PASS"
    )


if __name__ == "__main__":
    main()