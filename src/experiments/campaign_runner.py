"""
campaign_runner.py

Resumable campaign execution for the CA-SMRCPSP
multi-mode experimental protocol.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CampaignInstance:
    """
    One benchmark instance registered in the campaign manifest.
    """

    instance: str
    group: int
    replication: int
    path: Path


class CampaignRunner:
    """
    Resumable campaign runner.

    A campaign consists of:

        instance × seed × algorithm

    with three experimental algorithms:

        baseline_nsga2
        context_only_nsga2
        ca_nsga2

    This class is responsible for:

        1. Loading the campaign manifest.
        2. Validating manifest integrity.
        3. Providing structured campaign instances.

    Execution of optimization algorithms is deliberately
    implemented in a later stage.
    """

    ALGORITHMS = (
        "baseline_nsga2",
        "context_only_nsga2",
        "ca_nsga2",
    )

    MANIFEST_COLUMNS = (
        "instance",
        "group",
        "replication",
        "path",
    )

    def __init__(
        self,
        manifest: str | Path,
        seeds: Iterable[int],
        population_size: int = 20,
        generations: int = 20,
        result_root: str | Path = (
            "results/campaign/j30/raw"
        ),
    ) -> None:

        self.manifest = Path(manifest)

        self.seeds = [
            int(seed)
            for seed in seeds
        ]

        if not self.seeds:
            raise ValueError(
                "Seeds cannot be empty."
            )

        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "Seeds must be unique."
            )

        if population_size <= 0:
            raise ValueError(
                "Population size must be positive."
            )

        if generations <= 0:
            raise ValueError(
                "Generations must be positive."
            )

        self.population_size = int(
            population_size
        )

        self.generations = int(
            generations
        )

        self.result_root = Path(
            result_root
        )

        self.instances: list[CampaignInstance] = []

    # ---------------------------------------------------------
    # Manifest loading
    # ---------------------------------------------------------

    def load_manifest(
        self,
    ) -> list[CampaignInstance]:
        """
        Load and parse the campaign manifest.

        Returns
        -------
        list[CampaignInstance]
            Structured campaign instances.

        Raises
        ------
        FileNotFoundError
            If the manifest does not exist.

        ValueError
            If the manifest structure is invalid.
        """

        if not self.manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest}"
            )

        if not self.manifest.is_file():
            raise ValueError(
                f"Manifest is not a file: "
                f"{self.manifest}"
            )

        with self.manifest.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "Manifest has no header."
                )

            fieldnames = tuple(
                name.strip()
                for name in reader.fieldnames
                if name is not None
            )

            if fieldnames != self.MANIFEST_COLUMNS:
                raise ValueError(
                    "Invalid manifest columns. "
                    f"Expected {self.MANIFEST_COLUMNS}, "
                    f"found {fieldnames}."
                )

            records: list[CampaignInstance] = []

            for line_number, row in enumerate(
                reader,
                start=2,
            ):

                if not row:
                    continue

                instance = (
                    row["instance"] or ""
                ).strip()

                group_value = (
                    row["group"] or ""
                ).strip()

                replication_value = (
                    row["replication"] or ""
                ).strip()

                path_value = (
                    row["path"] or ""
                ).strip()

                if not instance:
                    raise ValueError(
                        f"Missing instance at "
                        f"manifest line {line_number}."
                    )

                if not group_value:
                    raise ValueError(
                        f"Missing group at "
                        f"manifest line {line_number}."
                    )

                if not replication_value:
                    raise ValueError(
                        f"Missing replication at "
                        f"manifest line {line_number}."
                    )

                if not path_value:
                    raise ValueError(
                        f"Missing path at "
                        f"manifest line {line_number}."
                    )

                try:
                    group = int(
                        group_value
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid group at "
                        f"manifest line {line_number}: "
                        f"{group_value}"
                    ) from exc

                try:
                    replication = int(
                        replication_value
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid replication at "
                        f"manifest line {line_number}: "
                        f"{replication_value}"
                    ) from exc

                if group <= 0:
                    raise ValueError(
                        f"Group must be positive at "
                        f"manifest line {line_number}."
                    )

                if replication <= 0:
                    raise ValueError(
                        f"Replication must be positive "
                        f"at manifest line "
                        f"{line_number}."
                    )

                records.append(
                    CampaignInstance(
                        instance=instance,
                        group=group,
                        replication=replication,
                        path=Path(path_value),
                    )
                )

        self.instances = records

        return list(self.instances)

    # ---------------------------------------------------------
    # Manifest validation
    # ---------------------------------------------------------

    def validate_manifest(
        self,
        instances: Iterable[CampaignInstance] | None = None,
    ) -> None:
        """
        Validate campaign manifest integrity.

        The validation checks:

            - non-empty manifest
            - unique instance names
            - unique group/replication combinations
            - existing benchmark files
            - positive group and replication IDs
            - exactly 10 replications per group
        """

        records = list(
            self.instances
            if instances is None
            else instances
        )

        if not records:
            raise ValueError(
                "Manifest contains no instances."
            )

        # -----------------------------------------------------
        # Unique instance names
        # -----------------------------------------------------

        instance_names = [
            record.instance
            for record in records
        ]

        if len(instance_names) != len(
            set(instance_names)
        ):
            raise ValueError(
                "Manifest contains duplicate "
                "instance names."
            )

        # -----------------------------------------------------
        # Unique group/replication combinations
        # -----------------------------------------------------

        group_replications = [
            (
                record.group,
                record.replication,
            )
            for record in records
        ]

        if len(group_replications) != len(
            set(group_replications)
        ):
            raise ValueError(
                "Manifest contains duplicate "
                "(group, replication) combinations."
            )

        # -----------------------------------------------------
        # Benchmark paths
        # -----------------------------------------------------

        missing_paths = [
            record.path
            for record in records
            if not record.path.exists()
        ]

        if missing_paths:
            preview = ", ".join(
                str(path)
                for path in missing_paths[:5]
            )

            raise FileNotFoundError(
                "Manifest contains missing "
                f"benchmark paths: {preview}"
            )

        # -----------------------------------------------------
        # Replication validation
        # -----------------------------------------------------

        groups: dict[
            int,
            list[CampaignInstance],
        ] = {}

        for record in records:

            groups.setdefault(
                record.group,
                [],
            ).append(record)

        invalid_groups = {
            group: len(group_records)
            for group, group_records
            in groups.items()
            if len(group_records) != 10
        }

        if invalid_groups:
            raise ValueError(
                "Each campaign group must contain "
                f"exactly 10 replications. "
                f"Invalid groups: "
                f"{invalid_groups}"
            )

    # ---------------------------------------------------------
    # Prepare campaign
    # ---------------------------------------------------------

    def prepare(
        self,
    ) -> list[CampaignInstance]:
        """
        Load and validate the campaign manifest.
        """

        instances = self.load_manifest()

        self.validate_manifest(
            instances
        )

        return list(
            self.instances
        )

    # ---------------------------------------------------------
    # Group access
    # ---------------------------------------------------------

    def groups(
        self,
    ) -> dict[int, list[CampaignInstance]]:
        """
        Return campaign instances grouped by group ID.
        """

        if not self.instances:
            self.prepare()

        grouped: dict[
            int,
            list[CampaignInstance],
        ] = {}

        for record in self.instances:

            grouped.setdefault(
                record.group,
                [],
            ).append(record)

        for records in grouped.values():

            records.sort(
                key=lambda record:
                record.replication
            )

        return dict(
            sorted(
                grouped.items()
            )
        )

    # ---------------------------------------------------------
    # Campaign summary
    # ---------------------------------------------------------

    @property
    def number_of_instances(
        self,
    ) -> int:

        return len(
            self.instances
        )

    @property
    def number_of_groups(
        self,
    ) -> int:

        return len(
            {
                record.group
                for record
                in self.instances
            }
        )

    @property
    def replications_per_group(
        self,
    ) -> int:

        groups = self.groups()

        if not groups:
            return 0

        counts = {
            len(records)
            for records
            in groups.values()
        }

        if len(counts) != 1:
            raise ValueError(
                "Campaign groups do not have "
                "uniform replication counts."
            )

        return counts.pop()

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, object]:

        if not self.instances:
            self.prepare()

        return {
            "manifest": str(
                self.manifest
            ),
            "instances": self.number_of_instances,
            "groups": self.number_of_groups,
            "replications_per_group":
                self.replications_per_group,
            "seeds": list(
                self.seeds
            ),
            "population_size":
                self.population_size,
            "generations":
                self.generations,
            "algorithms":
                list(
                    self.ALGORITHMS
                ),
            "result_root":
                str(
                    self.result_root
                ),
        }