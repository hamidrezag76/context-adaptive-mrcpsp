"""
validate_campaign_integrity.py

Integrity validation for the CA-SMRCPSP experimental campaign.

Validates:

    instance × seed × algorithm

against the campaign manifest and stored JSON results.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ALGORITHMS = (
    "baseline_nsga2",
    "context_only_nsga2",
    "ca_nsga2",
)


EXPECTED_OBJECTIVES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CA-SMRCPSP campaign result integrity."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/campaign/j30/manifest.csv"
        ),
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "results/campaign/j30/raw"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42, 52)),
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any integrity problem.",
    )

    return parser.parse_args()


def load_manifest(
    path: Path,
) -> list[dict[str, str]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise ValueError(
            "Manifest is empty."
        )

    required = {
        "instance",
        "group",
        "replication",
        "path",
    }

    actual = set(rows[0].keys())

    missing = required - actual

    if missing:
        raise AssertionError(
            "Manifest missing columns: "
            + ", ".join(sorted(missing))
        )

    return rows


def expected_keys(
    instances: list[str],
    seeds: list[int],
) -> set[tuple[str, int, str]]:

    return {
        (
            instance,
            int(seed),
            algorithm,
        )
        for instance in instances
        for seed in seeds
        for algorithm in ALGORITHMS
    }


def result_path(
    root: Path,
    instance: str,
    algorithm: str,
    seed: int,
) -> Path:

    return (
        root
        / Path(instance).stem
        / algorithm
        / f"seed_{int(seed)}.json"
    )


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise AssertionError(
            f"Result is not a JSON object: {path}"
        )

    return data


def validate_numeric(
    value: Any,
    name: str,
    path: Path,
) -> None:

    if isinstance(value, bool):
        raise AssertionError(
            f"{name} is boolean: {path}"
        )

    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ) as exc:

        raise AssertionError(
            f"{name} is not numeric: {path}"
        ) from exc

    if not math.isfinite(numeric):
        raise AssertionError(
            f"{name} is not finite: {path}"
        )


def validate_record(
    record: dict[str, Any],
    *,
    path: Path,
    instance: str,
    algorithm: str,
    seed: int,
    population_size: int,
    generations: int,
) -> None:

    required = {
        "instance",
        "algorithm",
        "seed",
        "population_size",
        "generations",
        "archive_size",
        "archive_objectives",
        "metrics",
        "best_objectives",
        "metadata",
    }

    missing = required - set(record.keys())

    if missing:
        raise AssertionError(
            f"Missing fields in {path}: "
            + ", ".join(sorted(missing))
        )

    if (
        Path(
            str(record["instance"])
        ).stem
        != Path(instance).stem
    ):

        raise AssertionError(
            f"Instance mismatch in {path}"
        )

    if record["algorithm"] != algorithm:

        raise AssertionError(
            f"Algorithm mismatch in {path}"
        )

    if int(record["seed"]) != int(seed):

        raise AssertionError(
            f"Seed mismatch in {path}"
        )

    if int(
        record["population_size"]
    ) != population_size:

        raise AssertionError(
            f"Population size mismatch in {path}"
        )

    if int(
        record["generations"]
    ) != generations:

        raise AssertionError(
            f"Generations mismatch in {path}"
        )

    archive = record[
        "archive_objectives"
    ]

    if not isinstance(
        archive,
        list,
    ):

        raise AssertionError(
            f"Archive is not a list: {path}"
        )

    if not archive:

        raise AssertionError(
            f"Empty archive: {path}"
        )

    archive_size = int(
        record["archive_size"]
    )

    if archive_size != len(archive):

        raise AssertionError(
            f"Archive size mismatch in {path}: "
            f"metadata={archive_size}, "
            f"actual={len(archive)}"
        )

    for index, point in enumerate(
        archive
    ):

        if not isinstance(
            point,
            list,
        ):

            raise AssertionError(
                f"Archive point {index} "
                f"is not a list: {path}"
            )

        if len(point) != EXPECTED_OBJECTIVES:

            raise AssertionError(
                f"Archive point {index} "
                f"has {len(point)} objectives "
                f"instead of "
                f"{EXPECTED_OBJECTIVES}: {path}"
            )

        for objective_index, value in enumerate(
            point
        ):

            validate_numeric(
                value,
                (
                    f"archive "
                    f"objective {objective_index}"
                ),
                path,
            )

    best = record[
        "best_objectives"
    ]

    if not isinstance(
        best,
        list,
    ):

        raise AssertionError(
            f"best_objectives is not a list: {path}"
        )

    if len(best) != EXPECTED_OBJECTIVES:

        raise AssertionError(
            f"best_objectives has "
            f"{len(best)} objectives "
            f"instead of "
            f"{EXPECTED_OBJECTIVES}: {path}"
        )

    for objective_index, value in enumerate(
        best
    ):

        validate_numeric(
            value,
            (
                f"best objective "
                f"{objective_index}"
            ),
            path,
        )

    metrics = record[
        "metrics"
    ]

    if not isinstance(
        metrics,
        dict,
    ):

        raise AssertionError(
            f"Metrics is not an object: {path}"
        )

    for metric_name in (
        "hypervolume",
        "igd_plus",
    ):

        if metric_name not in metrics:

            raise AssertionError(
                f"Missing metric "
                f"{metric_name}: {path}"
            )

        validate_numeric(
            metrics[metric_name],
            metric_name,
            path,
        )

        if float(
            metrics[metric_name]
        ) < 0.0:

            raise AssertionError(
                f"Negative metric "
                f"{metric_name}: {path}"
            )

    metadata = record[
        "metadata"
    ]

    if not isinstance(
        metadata,
        dict,
    ):

        raise AssertionError(
            f"Metadata is not an object: {path}"
        )

    expected_modes = {
        "baseline_nsga2": (
            "baseline",
            False,
            False,
        ),
        "context_only_nsga2": (
            "context_only",
            True,
            False,
        ),
        "ca_nsga2": (
            "full_ca",
            True,
            True,
        ),
    }

    expected_mode = expected_modes[
        algorithm
    ]

    if metadata.get(
        "mode"
    ) != expected_mode[0]:

        raise AssertionError(
            f"Mode metadata mismatch: {path}"
        )

    if bool(
        metadata.get(
            "context_adaptive"
        )
    ) != expected_mode[1]:

        raise AssertionError(
            f"context_adaptive metadata "
            f"mismatch: {path}"
        )

    if bool(
        metadata.get(
            "operator_adaptive"
        )
    ) != expected_mode[2]:

        raise AssertionError(
            f"operator_adaptive metadata "
            f"mismatch: {path}"
        )

    if (
        metadata.get(
            "metric_reference_set"
        )
        != "common_all_three_modes"
    ):

        raise AssertionError(
            f"Metric reference-set metadata "
            f"mismatch: {path}"
        )


def discover_result_files(
    root: Path,
) -> list[Path]:

    if not root.exists():
        return []

    return sorted(
        root.rglob(
            "seed_*.json"
        )
    )


def main() -> None:

    args = parse_args()

    manifest = load_manifest(
        args.manifest
    )

    instances = [
        row["instance"]
        for row in manifest
    ]

    instance_set = set(
        instances
    )

    if len(instance_set) != len(
        instances
    ):

        duplicates = [
            name
            for name, count
            in Counter(instances).items()
            if count > 1
        ]

        raise AssertionError(
            "Duplicate instances in manifest: "
            + ", ".join(
                sorted(duplicates)
            )
        )

    seeds = [
        int(seed)
        for seed in args.seeds
    ]

    if len(seeds) != len(set(seeds)):

        raise AssertionError(
            "Duplicate seeds supplied."
        )

    expected = expected_keys(
        sorted(instance_set),
        seeds,
    )

    files = discover_result_files(
        args.result_root
    )

    observed: set[
        tuple[str, int, str]
    ] = set()

    valid = 0
    invalid = 0

    errors: list[str] = []

    for path in files:

        try:

            record = load_json(
                path
            )

            instance = str(
                record["instance"]
            )

            algorithm = str(
                record["algorithm"]
            )

            seed = int(
                record["seed"]
            )

            key = (
                Path(instance).name,
                seed,
                algorithm,
            )

            if key in observed:

                raise AssertionError(
                    f"Duplicate result key: {key}"
                )

            observed.add(
                key
            )

            if (
                Path(instance).name
                not in instance_set
            ):

                raise AssertionError(
                    f"Instance not in manifest: "
                    f"{instance}"
                )

            if seed not in seeds:

                raise AssertionError(
                    f"Unexpected seed {seed}"
                    f": {path}"
                )

            if (
                algorithm
                not in ALGORITHMS
            ):

                raise AssertionError(
                    f"Unexpected algorithm "
                    f"{algorithm}: {path}"
                )

            validate_record(
                record,
                path=path,
                instance=instance,
                algorithm=algorithm,
                seed=seed,
                population_size=args.population_size,
                generations=args.generations,
            )

            valid += 1

        except Exception as exc:

            invalid += 1

            errors.append(
                f"{path}: {exc}"
            )

    missing = sorted(
        expected - observed
    )

    unexpected = sorted(
        observed - expected
    )

    total_expected = len(
        expected
    )

    total_observed = len(
        observed
    )

    completed = len(
        expected & observed
    )

    print(
        "\n========== CAMPAIGN INTEGRITY =========="
    )

    print(
        f"Manifest instances: {len(instance_set)}"
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"Algorithms: {ALGORITHMS}"
    )

    print(
        f"Expected runs: {total_expected}"
    )

    print(
        f"Observed runs: {total_observed}"
    )

    print(
        f"Valid runs: {valid}"
    )

    print(
        f"Invalid runs: {invalid}"
    )

    print(
        f"Completed expected runs: {completed}"
    )

    print(
        f"Missing runs: {len(missing)}"
    )

    print(
        f"Unexpected runs: {len(unexpected)}"
    )

    if missing:

        print(
            "\n========== MISSING RUNS =========="
        )

        for instance, seed, algorithm in (
            missing[:50]
        ):

            print(
                f"  {instance} "
                f"seed={seed} "
                f"{algorithm}"
            )

        if len(missing) > 50:

            print(
                f"  ... "
                f"{len(missing) - 50} more"
            )

    if unexpected:

        print(
            "\n========== UNEXPECTED RUNS =========="
        )

        for key in unexpected[:50]:

            print(
                " ",
                key,
            )

    if errors:

        print(
            "\n========== INVALID RESULTS =========="
        )

        for error in errors[:50]:

            print(
                " ",
                error,
            )

        if len(errors) > 50:

            print(
                f"  ... "
                f"{len(errors) - 50} more"
            )

    if (
        invalid == 0
        and not missing
        and not unexpected
        and total_observed == total_expected
    ):

        print(
            "\nCAMPAIGN INTEGRITY: PASS"
        )

    else:

        print(
            "\nCAMPAIGN INTEGRITY: INCOMPLETE"
        )

        if args.strict:

            raise SystemExit(1)


if __name__ == "__main__":
    main()