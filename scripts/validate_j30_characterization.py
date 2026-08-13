"""
validate_j30_characterization.py

Validation and variability analysis for the J30 benchmark
characterization dataset.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


INPUT = Path(
    "results/campaign/j30/characterization.csv"
)

OUTPUT = Path(
    "results/campaign/j30/characterization_validation.csv"
)


IDENTIFIER_COLUMNS = {
    "instance",
    "group",
    "replication",
    "path",
}


def read_characterization(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Characterization file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                "Characterization CSV has no header."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            "Characterization CSV is empty."
        )

    return rows, list(reader.fieldnames)


def numeric_columns(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> list[str]:

    columns = []

    for column in fieldnames:

        if column in IDENTIFIER_COLUMNS:
            continue

        valid = True

        for row in rows:

            value = row.get(column, "").strip()

            if value == "":
                valid = False
                break

            try:
                number = float(value)
            except ValueError:
                valid = False
                break

            if not math.isfinite(number):
                valid = False
                break

        if valid:
            columns.append(column)

    return columns


def variance(
    values: list[float],
) -> float:

    if len(values) <= 1:
        return 0.0

    return statistics.variance(values)


def coefficient_of_variation(
    mean: float,
    std: float,
) -> float:

    if mean == 0.0:
        return 0.0

    return abs(std / mean)


def validate_structure(
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:

    expected_instances = 640
    expected_groups = 64
    expected_replications = 10

    if len(rows) != expected_instances:

        raise AssertionError(
            f"Expected {expected_instances} instances, "
            f"found {len(rows)}."
        )

    groups = defaultdict(list)

    for row in rows:

        group = int(row["group"])

        replication = int(
            row["replication"]
        )

        groups[group].append(
            replication
        )

    if len(groups) != expected_groups:

        raise AssertionError(
            f"Expected {expected_groups} groups, "
            f"found {len(groups)}."
        )

    expected_group_ids = set(
        range(
            1,
            expected_groups + 1,
        )
    )

    if set(groups) != expected_group_ids:

        raise AssertionError(
            "Group IDs are not exactly 1..64."
        )

    for group, replications in groups.items():

        if len(replications) != expected_replications:

            raise AssertionError(
                f"Group {group} has "
                f"{len(replications)} replications; "
                f"expected {expected_replications}."
            )

        if set(replications) != set(
            range(
                1,
                expected_replications + 1,
            )
        ):

            raise AssertionError(
                f"Group {group} does not contain "
                "replications 1..10."
            )

    return (
        len(rows),
        len(groups),
        expected_replications,
    )


def validate_values(
    rows: list[dict[str, str]],
    numeric: list[str],
) -> None:

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        for column in numeric:

            value = row[column].strip()

            if value == "":
                raise AssertionError(
                    f"Missing value in "
                    f"{column}, CSV row {row_number}."
                )

            try:
                number = float(value)
            except ValueError as exc:
                raise AssertionError(
                    f"Non-numeric value in "
                    f"{column}, CSV row {row_number}: "
                    f"{value!r}"
                ) from exc

            if not math.isfinite(number):

                raise AssertionError(
                    f"NaN/Inf detected in "
                    f"{column}, CSV row {row_number}."
                )

def build_feature_statistics(
    rows: list[dict[str, str]],
    numeric: list[str],
) -> list[dict[str, object]]:

    groups = defaultdict(list)

    for row in rows:

        groups[int(row["group"])].append(
            row
        )

    results = []

    for feature in numeric:

        values = [
            float(row[feature])
            for row in rows
        ]

        mean = statistics.mean(
            values
        )

        std = (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        )

        minimum = min(values)

        maximum = max(values)

        unique_values = len(
            set(values)
        )

        # -------------------------------------------------
        # Group-level statistics
        # -------------------------------------------------

        group_means = []

        grouped_values = []

        for group_rows in groups.values():

            group_values = [
                float(row[feature])
                for row in group_rows
            ]

            grouped_values.append(
                group_values
            )

            group_means.append(
                statistics.mean(
                    group_values
                )
            )

        # -------------------------------------------------
        # ANOVA-style variance decomposition
        #
        # Total SS = Between SS + Within SS
        # -------------------------------------------------

        grand_mean = statistics.mean(
            values
        )

        group_sizes = [
            len(group_values)
            for group_values
            in grouped_values
        ]

        if len(
            set(group_sizes)
        ) != 1:

            raise AssertionError(
                "Unequal group sizes detected."
            )

        between_ss = 0.0

        within_ss = 0.0

        for group_values, group_mean in zip(
            grouped_values,
            group_means,
        ):

            group_size = len(
                group_values
            )

            between_ss += (
                group_size
                * (
                    group_mean
                    - grand_mean
                ) ** 2
            )

            within_ss += sum(
                (
                    value
                    - group_mean
                ) ** 2
                for value
                in group_values
            )

        total_ss = sum(
            (
                value
                - grand_mean
            ) ** 2
            for value in values
        )

        denominator = len(values) - 1

        if denominator > 0:

            between_group_variance = (
                between_ss
                / denominator
            )

            within_group_variance = (
                within_ss
                / denominator
            )

            total_variance = (
                total_ss
                / denominator
            )

        else:

            between_group_variance = 0.0

            within_group_variance = 0.0

            total_variance = 0.0

        # -------------------------------------------------
        # Variance shares
        # -------------------------------------------------

        if total_ss > 0.0:

            between_share = (
                between_ss
                / total_ss
            )

            within_share = (
                within_ss
                / total_ss
            )

            # -------------------------------------------------
            # Numerical validation
            # -------------------------------------------------

            if not (
                -1e-10
                <= between_share
                <= 1.0 + 1e-10
            ):

                raise AssertionError(
                    f"Invalid between-group variance "
                    f"share for {feature}: "
                    f"{between_share}"
                )

            if not (
                -1e-10
                <= within_share
                <= 1.0 + 1e-10
            ):

                raise AssertionError(
                    f"Invalid within-group variance "
                    f"share for {feature}: "
                    f"{within_share}"
                )

            if not math.isclose(
                between_share
                + within_share,
                1.0,
                rel_tol=1e-8,
                abs_tol=1e-8,
            ):

                raise AssertionError(
                    f"Variance decomposition does not "
                    f"sum to 1 for {feature}: "
                    f"{between_share + within_share}"
                )

        else:

            # Constant feature:
            # no variance exists, so the variance
            # decomposition shares are not defined.
            between_share = 0.0
            within_share = 0.0

        # -------------------------------------------------
        # Store feature statistics
        # -------------------------------------------------

        results.append(
            {
                "feature": feature,

                "mean": mean,

                "std": std,

                "min": minimum,

                "max": maximum,

                "cv":
                    coefficient_of_variation(
                        mean,
                        std,
                    ),

                "unique_values":
                    unique_values,

                "between_group_variance":
                    between_group_variance,

                "within_group_variance":
                    within_group_variance,

                "between_group_variance_share":
                    between_share,

                "within_group_variance_share":
                    within_share,
            }
        )

    return results


def write_statistics(
    path: Path,
    statistics_rows: list[dict[str, object]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not statistics_rows:
        raise ValueError(
            "No feature statistics available."
        )

    fieldnames = list(
        statistics_rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            statistics_rows
        )


def print_report(
    rows: list[dict[str, str]],
    numeric: list[str],
    statistics_rows: list[dict[str, object]],
) -> None:

    constant_features = [
        row["feature"]
        for row in statistics_rows
        if row["unique_values"] <= 1
    ]

    variable_features = [
        row["feature"]
        for row in statistics_rows
        if row["unique_values"] > 1
    ]

    highly_group_sensitive = sorted(
        statistics_rows,
        key=lambda row:
            float(
                row[
                    "between_group_variance_share"
                ]
            ),
        reverse=True,
    )

    highly_variable = sorted(
        statistics_rows,
        key=lambda row:
            float(row["cv"]),
        reverse=True,
    )

    print(
        "\n========== J30 CHARACTERIZATION VALIDATION =========="
    )

    print(
        "Instances:",
        len(rows),
    )

    print(
        "Numeric features:",
        len(numeric),
    )

    print(
        "Constant features:",
        len(constant_features),
    )

    print(
        "Variable features:",
        len(variable_features),
    )

    print(
        "\n========== CONSTANT FEATURES =========="
    )

    if constant_features:

        for feature in constant_features:
            print(
                f"  {feature}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\n========== TOP BETWEEN-GROUP FEATURES =========="
    )

    for row in highly_group_sensitive[:10]:

        print(
            f"  {row['feature']:<35} "
            f"between_share="
            f"{float(row['between_group_variance_share']):.4f}"
        )

    print(
        "\n========== TOP VARIABLE FEATURES =========="
    )

    for row in highly_variable[:10]:

        print(
            f"  {row['feature']:<35} "
            f"CV="
            f"{float(row['cv']):.4f}"
        )

    print(
        "\nValidation completed successfully."
    )


def main() -> None:

    rows, fieldnames = (
        read_characterization(
            INPUT
        )
    )

    (
        instance_count,
        group_count,
        replications_per_group,
    ) = validate_structure(
        rows
    )

    numeric = numeric_columns(
        rows,
        fieldnames,
    )

    if len(numeric) != 38:

        raise AssertionError(
            f"Expected 38 numeric characterization features, "
            f"found {len(numeric)}."
        )

    validate_values(
        rows,
        numeric,
    )

    statistics_rows = (
        build_feature_statistics(
            rows,
            numeric,
        )
    )

    write_statistics(
        OUTPUT,
        statistics_rows,
    )

    print(
        "J30 characterization validation "
        "completed successfully."
    )

    print(
        f"Instances: {instance_count}"
    )

    print(
        f"Groups: {group_count}"
    )

    print(
        "Replications per group:",
        replications_per_group,
    )

    print(
        "Total CSV columns:",
        len(fieldnames),
    )

    print(
        "Metadata columns:",
        len(
            IDENTIFIER_COLUMNS
        ),
    )

    print(
        "Numeric characterization features:",
        len(numeric),
    )

    print(
        "Statistics:",
        OUTPUT,
    )

    print_report(
        rows,
        numeric,
        statistics_rows,
    )


if __name__ == "__main__":
    main()