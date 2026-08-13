"""
build_j30_manifest.py

Builds a reproducible manifest for the complete J30 benchmark
inventory used in the CA-SMRCPSP experimental campaign.

Expected structure:

    j301_1.mm ... j301_10.mm
    ...
    j3064_1.mm ... j3064_10.mm

Total expected instances:
    64 groups × 10 replications = 640 instances.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


BENCHMARK_ROOT = Path("benchmarks/data")
OUTPUT_PATH = Path("results/campaign/j30/manifest.csv")

EXPECTED_GROUPS = 64
EXPECTED_REPLICATIONS = 10
EXPECTED_INSTANCES = (
    EXPECTED_GROUPS * EXPECTED_REPLICATIONS
)

PATTERN = re.compile(
    r"^j30(?P<group>\d+)_(?P<replication>\d+)\.mm$",
    re.IGNORECASE,
)


def discover_instances() -> list[dict[str, int | str]]:

    if not BENCHMARK_ROOT.exists():
        raise FileNotFoundError(
            f"Benchmark directory not found: {BENCHMARK_ROOT}"
        )

    records: list[dict[str, int | str]] = []

    for path in BENCHMARK_ROOT.glob("j30*.mm"):

        match = PATTERN.match(path.name)

        if match is None:
            continue

        group = int(
            match.group("group")
        )

        replication = int(
            match.group("replication")
        )

        records.append(
            {
                "instance": path.name,
                "group": group,
                "replication": replication,
                "path": str(path),
            }
        )

    return sorted(
        records,
        key=lambda record: (
            int(record["group"]),
            int(record["replication"]),
        ),
    )


def validate_inventory(
    records: list[dict[str, int | str]],
) -> None:

    if len(records) != EXPECTED_INSTANCES:
        raise ValueError(
            "Unexpected J30 instance count: "
            f"{len(records)}; expected "
            f"{EXPECTED_INSTANCES}."
        )

    groups = {
        int(record["group"])
        for record in records
    }

    expected_groups = set(
        range(1, EXPECTED_GROUPS + 1)
    )

    if groups != expected_groups:
        raise ValueError(
            "J30 group inventory is incomplete."
        )

    for group in range(
        1,
        EXPECTED_GROUPS + 1,
    ):

        group_records = [
            record
            for record in records
            if int(record["group"]) == group
        ]

        if len(group_records) != EXPECTED_REPLICATIONS:
            raise ValueError(
                f"Group j30{group} contains "
                f"{len(group_records)} instances; "
                f"expected {EXPECTED_REPLICATIONS}."
            )

        replications = {
            int(record["replication"])
            for record in group_records
        }

        expected_replications = set(
            range(1, EXPECTED_REPLICATIONS + 1)
        )

        if replications != expected_replications:
            raise ValueError(
                f"Group j30{group} has invalid "
                "replication inventory."
            )

    instance_names = [
        str(record["instance"])
        for record in records
    ]

    if len(instance_names) != len(
        set(instance_names)
    ):
        raise ValueError(
            "Duplicate instance names detected."
        )


def write_manifest(
    records: list[dict[str, int | str]],
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "instance",
                "group",
                "replication",
                "path",
            ],
        )

        writer.writeheader()

        writer.writerows(records)


def main() -> None:

    records = discover_instances()

    validate_inventory(records)

    write_manifest(records)

    print(
        "J30 manifest created successfully."
    )

    print(
        f"Groups: {EXPECTED_GROUPS}"
    )

    print(
        f"Replications per group: "
        f"{EXPECTED_REPLICATIONS}"
    )

    print(
        f"Total instances: {len(records)}"
    )

    print(
        f"Manifest: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()