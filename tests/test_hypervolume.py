from src.metrics.hypervolume import Hypervolume


def main():

    print("\n========== HYPERVOLUME TEST ==========")

    reference = (10.0, 10.0)

    hv = Hypervolume(reference)

    # Two non-dominated points.
    points = [
        (2.0, 8.0),
        (5.0, 5.0),
    ]

    value = hv.compute(points)

    print("Reference:", reference)
    print("Points:", points)
    print("HV:", value)

    # A dominated point must not change HV.
    points_with_dominated = [
        (2.0, 8.0),
        (5.0, 5.0),
        (7.0, 9.0),
    ]

    value_with_dominated = hv.compute(
        points_with_dominated
    )

    print(
        "HV with dominated point:",
        value_with_dominated
    )

    assert value == value_with_dominated

    # Better Pareto front should have larger HV.
    better = [
        (2.0, 6.0),
        (4.0, 4.0),
    ]

    better_value = hv.compute(
        better
    )

    print(
        "Better-front HV:",
        better_value
    )

    assert better_value > value

    print("\nHYPERVOLUME TEST: PASS")


if __name__ == "__main__":
    main()
