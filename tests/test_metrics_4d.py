from src.metrics.hypervolume import Hypervolume
from src.metrics.igd_plus import IGDPlus


def main():

    print()
    print("========== 4D METRICS VALIDATION ==========")

    # -------------------------------------------------
    # 4D Hypervolume
    # -------------------------------------------------

    reference = (
        1.0,
        1.0,
        1.0,
        1.0,
    )

    hv = Hypervolume(reference)

    # One point creates the complete unit hypercube.
    single_point = [
        (0.0, 0.0, 0.0, 0.0)
    ]

    value = hv.compute(single_point)

    print("Single-point 4D HV:", value)

    assert abs(value - 1.0) < 1e-10


    # -------------------------------------------------
    # Dominated point must not change HV
    # -------------------------------------------------

    points = [
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5),
    ]

    value_dominated = hv.compute(points)

    print(
        "4D HV with dominated point:",
        value_dominated,
    )

    assert abs(value_dominated - 1.0) < 1e-10


    # -------------------------------------------------
    # Two non-overlapping 4D regions
    # -------------------------------------------------

    points = [
    	(0.0, 0.5, 0.5, 0.5),
    	(0.5, 0.0, 0.5, 0.5),
        ]

    value_two = hv.compute(points)

    print(
        "Two-point 4D HV:",
        value_two,
    )

    assert abs(value_two - 0.1875) < 1e-10


    # -------------------------------------------------
    # IGD+
    # -------------------------------------------------

    reference_set = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
    ]

    igd = IGDPlus(reference_set)

    exact = igd.compute(
        reference_set
    )

    print(
        "Exact 4D IGD+:",
        exact,
    )

    assert abs(exact) < 1e-10


    # -------------------------------------------------
    # Worse approximation
    # -------------------------------------------------

    approximation = [
        (0.5, 0.5, 0.5, 0.5),
    ]

    worse = igd.compute(
        approximation
    )

    print(
        "Worse 4D IGD+:",
        worse,
    )

    assert worse > 0.0


    # -------------------------------------------------
    # Dimension mismatch
    # -------------------------------------------------

    try:

        hv.compute(
            [
                (0.0, 0.0, 0.0)
            ]
        )

    except ValueError:

        print(
            "HV dimension validation: PASS"
        )

    else:

        raise AssertionError(
            "HV accepted an invalid dimension."
        )


    try:

        igd.compute(
            [
                (0.0, 0.0, 0.0)
            ]
        )

    except ValueError:

        print(
            "IGD+ dimension validation: PASS"
        )

    else:

        raise AssertionError(
            "IGD+ accepted an invalid dimension."
        )


    print()
    print("4D METRICS VALIDATION: PASS")


if __name__ == "__main__":
    main()
