"""
result_store.py

Persistent storage for reproducible experimental results.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


class ResultStore:
    """
    Stores and retrieves reproducible optimization experiment results.

    Each run is stored independently using:

        results/raw/{instance}/{algorithm}/seed_{seed}.json
    """

    def __init__(
        self,
        root: str | Path = "results/raw",
    ) -> None:

        self.root = Path(root)

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # Path handling
    # ---------------------------------------------------------

    @staticmethod
    def _safe_name(value: str) -> str:

        value = str(value).strip()

        if not value:
            raise ValueError(
                "Name cannot be empty."
            )

        invalid = '<>:"/\\|?*'

        for character in invalid:
            value = value.replace(
                character,
                "_",
            )

        return value

    def run_path(
        self,
        instance: str,
        algorithm: str,
        seed: int,
    ) -> Path:

        instance_name = self._safe_name(
            Path(instance).stem
        )

        algorithm_name = self._safe_name(
            algorithm
        )

        return (
            self.root
            / instance_name
            / algorithm_name
            / f"seed_{int(seed)}.json"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save_run(
        self,
        *,
        instance: str,
        algorithm: str,
        seed: int,
        population_size: int,
        generations: int,
        archive_points: Iterable[Iterable[float]],
        metrics: dict[str, float],
        best_objectives: Iterable[float] | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:

        path = self.run_path(
            instance,
            algorithm,
            seed,
        )

        if path.exists() and not overwrite:

            raise FileExistsError(
                f"Result already exists: {path}"
            )

        archive_points = [
            [
                float(value)
                for value in point
            ]
            for point in archive_points
        ]

        if not archive_points:

            raise ValueError(
                "Archive points cannot be empty."
            )

        metrics = {
            str(key): float(value)
            for key, value in metrics.items()
        }

        record: dict[str, Any] = {

            "instance": str(instance),

            "algorithm": str(algorithm),

            "seed": int(seed),

            "population_size": int(
                population_size
            ),

            "generations": int(
                generations
            ),

            "archive_size": len(
                archive_points
            ),

            "archive_objectives":
                archive_points,

            "metrics":
                metrics,
        }

        if best_objectives is not None:

            record[
                "best_objectives"
            ] = [
                float(value)
                for value in best_objectives
            ]

        if metadata is not None:

            record[
                "metadata"
            ] = metadata

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                record,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            path
        )

        return path

    # ---------------------------------------------------------
    # Load one run
    # ---------------------------------------------------------

    def load_run(
        self,
        *,
        instance: str,
        algorithm: str,
        seed: int,
    ) -> dict[str, Any]:

        path = self.run_path(
            instance,
            algorithm,
            seed,
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Result not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # Load all runs
    # ---------------------------------------------------------

    def load_all(
        self,
    ) -> list[dict[str, Any]]:

        records = []

        for path in sorted(
            self.root.rglob("seed_*.json")
        ):

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                records.append(
                    json.load(file)
                )

        return records

    # ---------------------------------------------------------
    # Load filtered runs
    # ---------------------------------------------------------

    def find(
        self,
        *,
        instance: str | None = None,
        algorithm: str | None = None,
        seeds: Iterable[int] | None = None,
    ) -> list[dict[str, Any]]:

        records = self.load_all()

        seed_set = (
            {int(seed) for seed in seeds}
            if seeds is not None
            else None
        )

        result = []

        for record in records:

            if (
                instance is not None
                and Path(
                    record["instance"]
                ).stem
                != Path(instance).stem
            ):
                continue

            if (
                algorithm is not None
                and record["algorithm"]
                != algorithm
            ):
                continue

            if (
                seed_set is not None
                and record["seed"]
                not in seed_set
            ):
                continue

            result.append(record)

        return result

    # ---------------------------------------------------------
    # CSV export
    # ---------------------------------------------------------

    def export_csv(
        self,
        output_path: str | Path,
        records: Iterable[dict[str, Any]] | None = None,
    ) -> Path:

        if records is None:

            records = self.load_all()

        records = list(records)

        if not records:

            raise ValueError(
                "No records available for CSV export."
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "instance",
            "algorithm",
            "seed",
            "population_size",
            "generations",
            "archive_size",
            "best_makespan",
            "best_cost",
            "best_carbon",
            "best_energy",
            "hypervolume",
            "igd_plus",
        ]

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in records:

                metrics = record.get(
                    "metrics",
                    {},
                )

                best = record.get(
                    "best_objectives",
                    [None] * 4,
                )

                row = {

                    "instance":
                        record["instance"],

                    "algorithm":
                        record["algorithm"],

                    "seed":
                        record["seed"],

                    "population_size":
                        record["population_size"],

                    "generations":
                        record["generations"],

                    "archive_size":
                        record["archive_size"],

                    "best_makespan":
                        best[0],

                    "best_cost":
                        best[1],

                    "best_carbon":
                        best[2],

                    "best_energy":
                        best[3],

                    "hypervolume":
                        metrics.get(
                            "hypervolume"
                        ),

                    "igd_plus":
                        metrics.get(
                            "igd_plus"
                        ),
                }

                writer.writerow(row)

        return output_path
