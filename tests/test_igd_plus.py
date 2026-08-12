from src.metrics.igd_plus import IGDPlus


def main():

    print("\n========== IGD+ TEST ==========")

    reference = [
        (2.0, 8.0),
        (5.0, 5.0),
        (8.0, 2.0),
    ]

    metric = IGDPlus(
        reference
    )

    # Exact reproduction of the reference set.
    exact = metric.compute(
        reference
    )

    print(
        "Exact approximation IGD+:",
        exact
    )

    assert exact == 0.0

    # A worse approximation.
    worse = [
        (3.0, 9.0),
        (6.0, 6.0),
        (9.0, 3.0),
    ]

    worse_value = metric.compute(
        worse
    )

    print(
        "Worse approximation IGD+:",
        worse_value
    )

    assert worse_value > 0.0

    # A better/denser approximation should not
    # increase the distance to the reference set.
    dense = [
        (2.0, 8.0),
        (4.0, 6.0),
        (5.0, 5.0),
        (6.0, 4.0),
        (8.0, 2.0),
    ]

    dense_value = metric.compute(
        dense
    )

    print(
        "Dense approximation IGD+:",
        dense_value
    )

    assert dense_value <= worse_value

    print("\nIGD+ TEST: PASS")


if __name__ == "__main__":
    main()
