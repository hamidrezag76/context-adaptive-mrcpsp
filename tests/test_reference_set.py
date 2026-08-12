from src.experiments.reference_set import (
    ReferenceSetBuilder,
)


def main():

    print(
        "\n========== REFERENCE SET TEST =========="
    )

    baseline = [
        (0.10, 0.90),
        (0.30, 0.70),
        (0.80, 0.80),
    ]

    adaptive = [
        (0.20, 0.80),
        (0.40, 0.60),
        (0.90, 0.90),
    ]

    reference_set = (
        ReferenceSetBuilder.build(
            baseline,
            adaptive,
        )
    )

    print(
        "Reference set:",
        reference_set,
    )

    assert (
        (0.80, 0.80)
        not in reference_set
    )

    assert (
        (0.90, 0.90)
        not in reference_set
    )

    assert len(reference_set) == 4

    reference_point = (
        ReferenceSetBuilder.reference_point(
            reference_set,
            margin=0.05,
        )
    )

    print(
        "Reference point:",
        reference_point,
    )

    assert len(reference_point) == 2

    assert all(
        value > 0.0
        for value in reference_point
    )

    print(
        "Nondominated filtering: PASS"
    )

    print(
        "Reference point construction: PASS"
    )

    print(
        "REFERENCE SET TEST: PASS"
    )


if __name__ == "__main__":
    main()