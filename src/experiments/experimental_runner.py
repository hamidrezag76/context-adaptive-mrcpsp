"""
experimental_runner.py

Reproducible multi-seed experimental runner for
Baseline NSGA-II, Context-only NSGA-II,
and Full Context-Adaptive NSGA-II.

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
    Executes paired multi-seed experiments for three modes:

        1. Baseline:
           context_adaptive=False
           operator_adaptive=False

        2. Context-only:
           context_adaptive=True
           operator_adaptive=False

        3. Full CA:
           context_adaptive=True
           operator_adaptive=True

    The same instance, population size, generations,
    and random seeds are used across all three modes.
    """

    def __init__(
        self,
        instance: str | Path,
        seeds: Iterable[int],
        population_size: int = 20,
        generations: int = 20,
        result_root: str | Path = "results/raw",
    ) -> None:

        self.instance = Path(instance)

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

        # -----------------------------------------------------
        # Archives
        # -----------------------------------------------------

        self.baseline_archives = {}

        self.context_only_archives = {}

        self.adaptive_archives = {}

        # -----------------------------------------------------
        # Best solutions
        # -----------------------------------------------------

        self.baseline_best = {}

        self.context_only_best = {}

        self.adaptive_best = {}

        # -----------------------------------------------------
        # Archive sizes
        # -----------------------------------------------------

        self.baseline_archive_sizes = {}

        self.context_only_archive_sizes = {}

        self.adaptive_archive_sizes = {}

    # ---------------------------------------------------------
    # Run one algorithm
    # ---------------------------------------------------------

    def _run_algorithm(
        self,
        *,
        seed: int,
        context_adaptive: bool,
        operator_adaptive: bool,
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
            operator_adaptive=operator_adaptive,
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
                f"\nRunning seed {seed} ..."
            )

            # -------------------------------------------------
            # Baseline
            # -------------------------------------------------

            baseline_archive, baseline_best = (
                self._run_algorithm(
                    seed=seed,
                    context_adaptive=False,
                    operator_adaptive=False,
                )
            )

            # -------------------------------------------------
            # Context-only
            # -------------------------------------------------

            context_only_archive, context_only_best = (
                self._run_algorithm(
                    seed=seed,
                    context_adaptive=True,
                    operator_adaptive=False,
                )
            )

            # -------------------------------------------------
            # Full CA
            # -------------------------------------------------

            adaptive_archive, adaptive_best = (
                self._run_algorithm(
                    seed=seed,
                    context_adaptive=True,
                    operator_adaptive=True,
                )
            )

            # -------------------------------------------------
            # Store archives
            # -------------------------------------------------

            self.baseline_archives[
                seed
            ] = baseline_archive

            self.context_only_archives[
                seed
            ] = context_only_archive

            self.adaptive_archives[
                seed
            ] = adaptive_archive

            # -------------------------------------------------
            # Store best solutions
            # -------------------------------------------------

            self.baseline_best[
                seed
            ] = baseline_best

            self.context_only_best[
                seed
            ] = context_only_best

            self.adaptive_best[
                seed
            ] = adaptive_best

            # -------------------------------------------------
            # Store archive sizes
            # -------------------------------------------------

            self.baseline_archive_sizes[
                seed
            ] = len(
                baseline_archive
            )

            self.context_only_archive_sizes[
                seed
            ] = len(
                context_only_archive
            )

            self.adaptive_archive_sizes[
                seed
            ] = len(
                adaptive_archive
            )

            print(
                f"  Baseline archive:    "
                f"{len(baseline_archive)}"
            )

            print(
                f"  Context-only archive:"
                f" {len(context_only_archive)}"
            )

            print(
                f"  Full CA archive:     "
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

        if not self.context_only_archives:

            raise RuntimeError(
                "Context-only experiments "
                "have not been executed."
            )

        if not self.adaptive_archives:

            raise RuntimeError(
                "Full CA experiments "
                "have not been executed."
            )

        return MultiSeedMetricsEvaluator(
            self.baseline_archives,
            self.context_only_archives,
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
            int(result["seed"]): result
            for result in evaluator.evaluate()
        }

        for seed in self.seeds:

            metric = per_seed_metrics[
                seed
            ]

            # -------------------------------------------------
            # Baseline metrics
            # -------------------------------------------------

            baseline_metrics = {
                "hypervolume":
                    metric["baseline_hv"],

                "igd_plus":
                    metric["baseline_igd_plus"],
            }

            # -------------------------------------------------
            # Context-only metrics
            # -------------------------------------------------

            context_only_metrics = {
                "hypervolume":
                    metric["context_only_hv"],

                "igd_plus":
                    metric["context_only_igd_plus"],
            }

            # -------------------------------------------------
            # Full CA metrics
            # -------------------------------------------------

            adaptive_metrics = {
                "hypervolume":
                    metric["adaptive_hv"],

                "igd_plus":
                    metric["adaptive_igd_plus"],
            }

            # -------------------------------------------------
            # Common metadata
            # -------------------------------------------------

            common_metadata = {
                "metric_reference_set":
                    "common_all_three_modes",

                "metric_reference_point":
                    list(
                        evaluator.reference_point
                    ),

                "reference_set_size":
                    len(
                        evaluator.reference_set
                    ),
            }

            # -------------------------------------------------
            # Baseline
            # -------------------------------------------------

            baseline_metadata = {
                **common_metadata,

                "mode":
                    "baseline",

                "context_adaptive":
                    False,

                "operator_adaptive":
                    False,
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
                metadata=baseline_metadata,
                overwrite=True,
            )

            # -------------------------------------------------
            # Context-only
            # -------------------------------------------------

            context_only_metadata = {
                **common_metadata,

                "mode":
                    "context_only",

                "context_adaptive":
                    True,

                "operator_adaptive":
                    False,
            }

            self.result_store.save_run(
                instance=self.instance.name,
                algorithm="context_only_nsga2",
                seed=seed,
                population_size=self.population_size,
                generations=self.generations,
                archive_points=(
                    self.context_only_archives[seed]
                ),
                metrics=context_only_metrics,
                best_objectives=(
                    self.context_only_best[seed]
                ),
                metadata=context_only_metadata,
                overwrite=True,
            )

            # -------------------------------------------------
            # Full CA
            # -------------------------------------------------

            adaptive_metadata = {
                **common_metadata,

                "mode":
                    "full_ca",

                "context_adaptive":
                    True,

                "operator_adaptive":
                    True,
            }

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
                metadata=adaptive_metadata,
                overwrite=True,
            )

    # ---------------------------------------------------------
    # Complete experiment
    # ---------------------------------------------------------

    def run(
        self,
    ) -> dict:

        print(
            "\n========== THREE-MODE EXPERIMENT =========="
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

        print(
            "\nModes:"
        )

        print(
            "  1. Baseline     "
            "(context=False, operator=False)"
        )

        print(
            "  2. Context-only "
            "(context=True, operator=False)"
        )

        print(
            "  3. Full CA      "
            "(context=True, operator=True)"
        )

        self.run_algorithms()

        evaluator = self.evaluate_metrics()

        self.store_results(
            evaluator
        )

        summary = evaluator.summary()

        # -----------------------------------------------------
        # Hypervolume summary
        # -----------------------------------------------------

        print(
            "\n========== HYPERVOLUME SUMMARY =========="
        )

        print(
            "Baseline mean:",
            summary[
                "hypervolume"
            ][
                "baseline_mean"
            ],
        )

        print(
            "Context-only mean:",
            summary[
                "hypervolume"
            ][
                "context_only_mean"
            ],
        )

        print(
            "Full CA mean:",
            summary[
                "hypervolume"
            ][
                "adaptive_mean"
            ],
        )

        print(
            "Context-only vs Baseline:",
            summary[
                "hypervolume"
            ][
                "context_only_improvement_percent"
            ],
        )

        print(
            "Full CA vs Context-only:",
            summary[
                "hypervolume"
            ][
                "adaptive_vs_context_only_improvement_percent"
            ],
        )

        print(
            "Full CA vs Baseline:",
            summary[
                "hypervolume"
            ][
                "adaptive_improvement_percent"
            ],
        )

        # -----------------------------------------------------
        # IGD+ summary
        # -----------------------------------------------------

        print(
            "\n========== IGD+ SUMMARY =========="
        )

        print(
            "Baseline mean:",
            summary[
                "igd_plus"
            ][
                "baseline_mean"
            ],
        )

        print(
            "Context-only mean:",
            summary[
                "igd_plus"
            ][
                "context_only_mean"
            ],
        )

        print(
            "Full CA mean:",
            summary[
                "igd_plus"
            ][
                "adaptive_mean"
            ],
        )

        print(
            "Context-only vs Baseline:",
            summary[
                "igd_plus"
            ][
                "context_only_improvement_percent"
            ],
        )

        print(
            "Full CA vs Context-only:",
            summary[
                "igd_plus"
            ][
                "adaptive_vs_context_only_improvement_percent"
            ],
        )

        print(
            "Full CA vs Baseline:",
            summary[
                "igd_plus"
            ][
                "adaptive_improvement_percent"
            ],
        )

        return summary
