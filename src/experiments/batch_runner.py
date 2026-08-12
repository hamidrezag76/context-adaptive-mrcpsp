"""
batch_runner.py

Batch experimental runner for multiple PSPLIB instances.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.experiments.experimental_runner import (
    ExperimentalRunner,
)


class BatchExperimentRunner:
    """
    Executes ExperimentalRunner independently
    for multiple benchmark instances.
    """

    def __init__(
        self,
        instances: Iterable[str | Path],
        seeds: Iterable[int],
        population_size: int = 20,
        generations: int = 20,
        result_root: str | Path = "results/raw",
    ) -> None:

        self.instances = [
            Path(instance)
            for instance in instances
        ]

        self.seeds = [
            int(seed)
            for seed in seeds
        ]

        if not self.instances:
            raise ValueError(
                "Instances cannot be empty."
            )

        if not self.seeds:
            raise ValueError(
                "Seeds cannot be empty."
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

        self.results = {}

    # ---------------------------------------------------------
    # Run one instance
    # ---------------------------------------------------------

    def run_instance(
        self,
        instance: Path,
    ) -> dict:

        if not instance.exists():
            raise FileNotFoundError(
                f"Instance not found: {instance}"
            )

        print()
        print(
            "================================================="
        )
        print(
            "INSTANCE:",
            instance.name,
        )
        print(
            "================================================="
        )

        runner = ExperimentalRunner(
            instance=instance,
            seeds=self.seeds,
            population_size=self.population_size,
            generations=self.generations,
            result_root=self.result_root,
        )

        summary = runner.run()

        self.results[
            instance.name
        ] = {
            "summary": summary,
            "runner": runner,
        }

        return summary

    # ---------------------------------------------------------
    # Run all instances
    # ---------------------------------------------------------

    def run(
        self,
    ) -> dict[str, dict]:

        print()
        print(
            "#################################################"
        )
        print(
            "BATCH EXPERIMENT"
        )
        print(
            "#################################################"
        )

        print(
            "Instances:",
            len(self.instances),
        )

        print(
            "Seeds:",
            self.seeds,
        )

        print(
            "Population:",
            self.population_size,
        )

        print(
            "Generations:",
            self.generations,
        )

        for instance in self.instances:

            self.run_instance(
                instance
            )

        return {
            name: data["summary"]
            for name, data
            in self.results.items()
        }

    # ---------------------------------------------------------
    # Access results
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, dict]:

        return {
            name: data["summary"]
            for name, data
            in self.results.items()
        }

    def get_runner(
        self,
        instance_name: str,
    ) -> ExperimentalRunner:

        if instance_name not in self.results:
            raise KeyError(
                f"Instance has not been executed: "
                f"{instance_name}"
            )

        return self.results[
            instance_name
        ]["runner"]
