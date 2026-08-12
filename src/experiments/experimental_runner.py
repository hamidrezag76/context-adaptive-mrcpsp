"""
experimental_runner.py

Reproducible multi-seed experimental runner for
Baseline NSGA-II and Context-Adaptive NSGA-II.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2

from src.experiments.metrics_evaluator import (
    MultiSeedMetricsEvaluator,
)

from src.experiments.result_store import (
    ResultStore,
)


class ExperimentalRunner:
    """
    Executes paired baseline and context-adaptive
    experiments across a common set of random seeds.

    The same instance, population size, generations,
    and seed are used for both algorithms.
    """

    def __init__(
        self,
        instance: str | Path,
        seeds: Iterable[int],
        population_size: int = 20,
        generations: int = 20,
        result_root: str | Path = "results/raw",
    ) -> None:

        self.instance = Path(
            instance
        )

        self.seeds = [
            int(seed)
            for seed in seeds
        ]

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

        self.result_store = ResultStore(
            result_root
        )

        self.baseline_archives = {}
        self.adaptive_archives = {}

        self.baseline_best = {}
        self.adaptive_best = {}

        self.baseline_archive_sizes = {}
        self.adaptive_archive_sizes = {}

    # ---------------------------------------------------------
    # Run one algorithm
    # ---------------------------------------------------------

    def _run_algorithm(
        self,
        *,
        seed: int,
        context_adaptive: bool,
    ):

        project = MMParser(
            self.instance
        ).parse()

        algorithm = NSGA2(
            project,
            population_size=self.population_size,
            generations=self.generations,
            seed=seed,
            context_adaptive=context_adaptive,
        )

        algorithm.run()

        archive_points = [
            tuple(
                float(value)
                for value in chromosome.objectives
            )
            for chromosome
            in algorithm.archive.archive
        ]

        if not archive_points:
            raise RuntimeError(
                "Algorithm produced an empty archive."
            )

        best = min(
            algorithm.population.individuals,
            key=lambda chromosome:
                chromosome.makespan,
        )

        return (
            archive_points,
            tuple(
                float(value)
                for value in best.objectives
            ),
        )

    # ---------------------------------------------------------
    # Execute all seeds
    # ---------------------------------------------------------

    def run_algorithms(self) -> None:

        for seed in self.seeds:

            print(
                f"Running seed {seed} ..."
            )

            baseline_archive, baseline_best = (
                self._run_algorithm(
                    seed=seed,
                    context_adaptive=False,
                )
            )

            adaptive_archive, adaptive_best = (
                self._run_algorithm(
                    seed=seed,
                    context_adaptive=True,
                )
            )

            self.baseline_archives[
                seed
            ] = baseline_archive

            self.adaptive_archives[
                seed
            ] = adaptive_archive

            self.baseline_best[
                seed
            ] = baseline_best

            self.adaptive_best[
                seed
            ] = adaptive_best

            self.baseline_archive_sizes[
                seed
            ] = len(
                baseline_archive
            )

            self.adaptive_archive_sizes[
                seed
            ] = len(
                adaptive_archive
            )

            print(
                f"  Baseline archive: "
                f"{len(baseline_archive)}"
            )

            print(
                f"  CA archive:       "
                f"{len(adaptive_archive)}"
            )

    # ---------------------------------------------------------
    # Evaluate common metrics
    # ---------------------------------------------------------

    def evaluate_metrics(
        self,
    ) -> MultiSeedMetricsEvaluator:

        if not self.baseline_archives:
            raise RuntimeError(
                "Experiments have not been executed."
            )

        return MultiSeedMetricsEvaluator(
            self.baseline_archives,
            self.adaptive_archives,
        )

    # ---------------------------------------------------------
    # Store all results
    # ---------------------------------------------------------

    def store_results(
        self,
        evaluator: MultiSeedMetricsEvaluator,
    ) -> None:

        per_seed_metrics = {
            result["seed"]: result
            for result in evaluator.evaluate()
        }

        for seed in self.seeds:

            metric = per_seed_metrics[
                seed
            ]

            baseline_metrics = {
                "hypervolume":
                    metric["baseline_hv"],

                "igd_plus":
                    metric["baseline_igd_plus"],
            }

            adaptive_metrics = {
                "hypervolume":
                    metric["adaptive_hv"],

                "igd_plus":
                    metric["adaptive_igd_plus"],
            }

            self.result_store.save_run(
                instance=self.instance.name,
                algorithm="baseline_nsga2",
                seed=seed,
                population_size=self.population_size,
                generations=self.generations,
                archive_points=(
                    self.baseline_archives[seed]
                ),
                metrics=baseline_metrics,
                best_objectives=(
                    self.baseline_best[seed]
                ),
                metadata={
                    "context_adaptive": False,
                    "metric_reference_set":
                        "common_all_seeds",
                    "metric_reference_point":
                        list(
                            evaluator.reference_point
                        ),
                    "reference_set_size":
                        len(
                            evaluator.reference_set
                        ),
                },
                overwrite=True,
            )

            self.result_store.save_run(
                instance=self.instance.name,
                algorithm="ca_nsga2",
                seed=seed,
                population_size=self.population_size,
                generations=self.generations,
                archive_points=(
                    self.adaptive_archives[seed]
                ),
                metrics=adaptive_metrics,
                best_objectives=(
                    self.adaptive_best[seed]
                ),
                metadata={
                    "context_adaptive": True,
                    "metric_reference_set":
                        "common_all_seeds",
                    "metric_reference_point":
                        list(
                            evaluator.reference_point
                        ),
                    "reference_set_size":
                        len(
                            evaluator.reference_set
                        ),
                },
                overwrite=True,
            )

    # ---------------------------------------------------------
    # Complete experiment
    # ---------------------------------------------------------

    def run(
        self,
    ) -> dict:

        print(
            "\n========== EXPERIMENT RUN =========="
        )

        print(
            "Instance:",
            self.instance.name,
        )

        print(
            "Population:",
            self.population_size,
        )

        print(
            "Generations:",
            self.generations,
        )

        print(
            "Seeds:",
            self.seeds,
        )

        self.run_algorithms()

        evaluator = self.evaluate_metrics()

        self.store_results(
            evaluator
        )

        summary = evaluator.summary()

        print(
            "\n========== METRIC SUMMARY =========="
        )

        print(
            "HV baseline mean:",
            summary[
                "hypervolume"
            ][
                "baseline_mean"
            ],
        )

        print(
            "HV CA mean:",
            summary[
                "hypervolume"
            ][
                "adaptive_mean"
            ],
        )

        print(
            "HV improvement:",
            summary[
                "hypervolume"
            ][
                "relative_improvement_percent"
            ],
        )

        print(
            "IGD+ baseline mean:",
            summary[
                "igd_plus"
            ][
                "baseline_mean"
            ],
        )

        print(
            "IGD+ CA mean:",
            summary[
                "igd_plus"
            ][
                "adaptive_mean"
            ],
        )

        print(
            "IGD+ improvement:",
            summary[
                "igd_plus"
            ][
                "relative_improvement_percent"
            ],
        )

        return summary
