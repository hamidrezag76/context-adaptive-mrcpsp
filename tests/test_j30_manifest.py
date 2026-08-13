from __future__ import annotations

import csv
from pathlib import Path


MANIFEST = Path(
    "results/campaign/j30/manifest.csv"
)

EXPECTED_GROUPS = 64
EXPECTED_REPLICATIONS = 10
EXPECTED_INSTANCES = 640


def test_j30_manifest_exists():
    assert MANIFEST.exists()


def test_j30_manifest_inventory():

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == EXPECTED_INSTANCES

    groups = {
        int(row["group"])
        for row in rows
    }

    assert groups == set(
        range(
            1,
            EXPECTED_GROUPS + 1,
        )
    )

    for group in range(
        1,
        EXPECTED_GROUPS + 1,
    ):

        group_rows = [
            row
            for row in rows
            if int(row["group"]) == group
        ]

        assert len(group_rows) == (
            EXPECTED_REPLICATIONS
        )

        replications = {
            int(row["replication"])
            for row in group_rows
        }

        assert replications == set(
            range(
                1,
                EXPECTED_REPLICATIONS + 1,
            )
        )


def test_j30_manifest_unique_instances():

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    instances = [
        row["instance"]
        for row in rows
    ]

    assert len(instances) == len(
        set(instances)
    )


def test_j30_manifest_paths_exist():

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    for row in rows:

        path = Path(
            row["path"]
        )

        assert path.exists(), (
            f"Missing benchmark: {path}"
        )


def test_j30_manifest_order():

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    assert rows[0]["instance"] == (
        "j301_1.mm"
    )

    assert rows[-1]["instance"] == (
        "j3064_10.mm"
    )

    for index, row in enumerate(rows):

        expected_group = (
            index // EXPECTED_REPLICATIONS
        ) + 1

        expected_replication = (
            index % EXPECTED_REPLICATIONS
        ) + 1

        assert int(
            row["group"]
        ) == expected_group

        assert int(
            row["replication"]
        ) == expected_replication