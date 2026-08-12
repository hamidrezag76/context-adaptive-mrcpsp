"""
benchmark_runner.py

Runs benchmark experiments for CA-NSGA-II.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations
from src.optimization.nsga2.fast_non_dominated_sort import (
    FastNonDominatedSort,
)
import csv
import time
from pathlib import Path

import numpy as np

from src.evaluation.performance_metrics import PerformanceMetrics
from src.optimization.ca_nsga2 import NSGA2
from src.parser.mm_parser import MMParser
from src.evaluation.reference_front import ReferenceFrontBuilder


class BenchmarkRunner:
    """
    Execute benchmark experiments over PSPLIB instances.
    """

    def __init__(
        self,
        benchmark_directory: Path,
        output_directory: Path,
    ) -> None:

        self.benchmark_directory = benchmark_directory
        self.output_directory = output_directory

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.parser = MMParser

    def run(
        self,
        population_size: int = 100,
        generations: int = 80,
        pattern: str = "*.mm",
        start: int = 0,
        limit: int | None = None,
        repetitions: int = 30,
        seed_start: int = 1,
    ) -> None:

        result_file = self.output_directory / "benchmark_results.csv"
        
        file_exists = result_file.exists()

        with result_file.open(
            "a",
            newline="",
        ) as file:

            writer = csv.writer(file)
            
            if not file_exists:

                writer.writerow(
                    [
                        "Instance",
                        "Activities",
                        "Population",
                        "Generations",
                        "ParetoSolutions",
                        "BestMakespan",
                        "BestCost",
                        "BestCarbon",
                        "BestEnergy",
                        "CA_HV",
                        "CA_HV_STD",
                        "Baseline_HV",
                        "Baseline_HV_STD",
                        "HV_Improvement",
                        "CA_IGD",
                        "CA_IGD_STD",
                        "Baseline_IGD",
                        "Baseline_IGD_STD",
                        "IGD_Improvement",
                        "CA_Spacing",
                        "CA_Spacing_STD",
                        "Baseline_Spacing",
                        "Baseline_Spacing_STD",
                        "Spacing_Improvement",
                        "CA_Spread",
                        "CA_Spread_STD",
                        "Baseline_Spread",
                        "Baseline_Spread_STD",
                        "Spread_Improvement",
                        "MeanTime",
                        "StdTime",
                        "Status",
                    ]
                )

            benchmark_files = sorted(
                self.benchmark_directory.glob(pattern)
            )

            benchmark_files = benchmark_files[start:]

            if limit is not None:
                benchmark_files = benchmark_files[:limit]

            for index, benchmark in enumerate(
                benchmark_files,
                start=1,
            ):

                try:

                    print(f"[{index}/{len(benchmark_files)}] " f"{benchmark.name}")

                    project = self.parser(
                        benchmark,
                    ).parse()

                    metrics = PerformanceMetrics()
                    reference_builder = ReferenceFrontBuilder()

                    ca_fronts = []
                    ca_runs = []

                    baseline_fronts = []
                    baseline_runs = []

                    execution_times = []

                    for run_id in range(repetitions):

                        optimizer = NSGA2(
                            project=project,
                            population_size=population_size,
                            generations=generations,
                            seed=seed_start + run_id,
                        )

                        start = time.perf_counter()

                        population = optimizer.run()
                        
                        reference_builder.add_population(
                            population.individuals
                        )

                        sorter = FastNonDominatedSort()

                        fronts = sorter.sort(
                            population.individuals,
                        )

                        pareto = fronts[0]

                        baseline_optimizer = NSGA2(
                            project=project,
                            population_size=population_size,
                            generations=generations,
                            seed=seed_start + run_id,
                            context_adaptive=False,
                        )

                        baseline_population = baseline_optimizer.run()
                        
                        reference_builder.add_population(
                            baseline_population.individuals
                        )

                        baseline_sorted_fronts = sorter.sort(
                            baseline_population.individuals,
                        )

                        baseline_pareto = baseline_sorted_fronts[0]

                        elapsed = time.perf_counter() - start

                        ca_front = np.array(
                            [
                                [
                                    c.makespan,
                                    c.total_cost,
                                    c.total_carbon,
                                    c.total_energy,
                                ]
                                for c in pareto
                            ]
                        )

                        baseline_front = np.array(
                            [
                                [
                                    c.makespan,
                                    c.total_cost,
                                    c.total_carbon,
                                    c.total_energy,
                                ]
                                for c in baseline_pareto
                            ]
                        )

                        ca_fronts.append(ca_front)
                        
                        ca_runs.append(
                            [
                                c.copy()
                                for c in pareto
                            ]
                        )

                        baseline_fronts.append(baseline_front)
                        
                        baseline_runs.append(
                            [
                                c.copy()
                                for c in baseline_pareto
                            ]
                        )

                        execution_times.append(elapsed)
                        # ------------------------------------------
                    # Reference Front
                    # ------------------------------------------

                    reference_builder = ReferenceFrontBuilder()

                    for pareto in ca_runs:
                        reference_builder.add(pareto)
                        reference_builder.add_population(pareto)

                    for pareto in baseline_runs:
                        reference_builder.add(pareto)
                        reference_builder.add_population(pareto)

                    # ------------------------------------------
                    # Reference Front and Common Normalization
                    # ------------------------------------------

                    reference_front_raw = (
                        reference_builder.build()
                    )

                    minimum, denominator = (
                        reference_builder.normalization_bounds()
                    )

                    reference_front = (
                        reference_builder.normalize_external(
                            reference_front_raw,
                            minimum,
                            denominator,
                        )
                    )

                    # ------------------------------------------
                    # Containers
                    # ------------------------------------------

                    ca_hv_values = []
                    ca_igd_values = []
                    ca_spacing_values = []
                    ca_spread_values = []

                    baseline_hv_values = []
                    baseline_igd_values = []
                    baseline_spacing_values = []
                    baseline_spread_values = []

                    best_hv = float("-inf")
                    best_pareto = None

                    # ------------------------------------------
                    # Context-Adaptive Metrics
                    # ------------------------------------------

                    for front, pareto in zip(
                        ca_fronts,
                        ca_runs,
                    ):
                        
                        front = np.array(

                            [

                                [

                                    c.makespan,

                                    c.total_cost,

                                    c.total_carbon,

                                    c.total_energy,

                                ]

                                for c in pareto

                            ]

                        )
                        

                        normalized = reference_builder.normalize_external(
                            front,
                            minimum,
                            denominator,
                        )

                        reference_point = np.ones(4) * 1.1

                        hv = metrics.hypervolume(
                            normalized,
                            reference_point,
                        )

                        igd = metrics.inverted_generational_distance(
                            normalized,
                            reference_front,
                        )

                        spacing = metrics.spacing(
                            normalized,
                        )

                        spread = metrics.spread(
                            pareto,
                        )

                        ca_hv_values.append(hv)
                        ca_igd_values.append(igd)
                        ca_spacing_values.append(spacing)
                        ca_spread_values.append(spread)
                        
                        if hv > best_hv:

                            best_hv = hv

                            best_pareto = [
                                c.copy()
                                for c in pareto
                            ]

                    # ------------------------------------------
                    # Baseline Metrics
                    # ------------------------------------------

                    for front, pareto in zip(
                        baseline_fronts,
                        baseline_runs,
                    ):
                        
                        front = np.array(

                            [

                                [

                                    c.makespan,

                                    c.total_cost,

                                    c.total_carbon,

                                    c.total_energy,

                                ]

                                for c in pareto

                            ]

                        )

                        normalized = reference_builder.normalize_external(
                            front,
                            minimum,
                            denominator,
                        )

                        reference_point = np.ones(4) * 1.1

                        hv = metrics.hypervolume(
                            normalized,
                            reference_point,
                        )

                        igd = metrics.inverted_generational_distance(
                            normalized,
                            reference_front,
                        )

                        spacing = metrics.spacing(
                            normalized,
                        )

                        spread = metrics.spread(
                            pareto,
                        )

                        baseline_hv_values.append(hv)
                        baseline_igd_values.append(igd)
                        baseline_spacing_values.append(spacing)
                        baseline_spread_values.append(spread)
                        # ------------------------------------------
                    # Mean / Std
                    # ------------------------------------------

                    ca_mean_hv = float(np.mean(ca_hv_values))
                    ca_std_hv = float(np.std(ca_hv_values))

                    ca_mean_igd = float(np.mean(ca_igd_values))
                    ca_std_igd = float(np.std(ca_igd_values))

                    ca_mean_spacing = float(np.mean(ca_spacing_values))
                    ca_std_spacing = float(np.std(ca_spacing_values))

                    ca_mean_spread = float(np.mean(ca_spread_values))
                    ca_std_spread = float(np.std(ca_spread_values))

                    baseline_mean_hv = float(np.mean(baseline_hv_values))
                    baseline_std_hv = float(np.std(baseline_hv_values))

                    baseline_mean_igd = float(np.mean(baseline_igd_values))
                    baseline_std_igd = float(np.std(baseline_igd_values))

                    baseline_mean_spacing = float(np.mean(baseline_spacing_values))
                    baseline_std_spacing = float(np.std(baseline_spacing_values))

                    baseline_mean_spread = float(np.mean(baseline_spread_values))
                    baseline_std_spread = float(np.std(baseline_spread_values))

                    mean_time = float(np.mean(execution_times))

                    std_time = float(np.std(execution_times))

                    # ------------------------------------------
                    # Improvements
                    # ------------------------------------------

                    hv_improvement = (
                        (ca_mean_hv - baseline_mean_hv) / baseline_mean_hv * 100
                        if baseline_mean_hv > 0
                        else 0.0
                    )

                    igd_improvement = (
                        (baseline_mean_igd - ca_mean_igd) / baseline_mean_igd * 100
                        if baseline_mean_igd > 0
                        else 0.0
                    )

                    spacing_improvement = (
                        (baseline_mean_spacing - ca_mean_spacing)
                        / baseline_mean_spacing
                        * 100
                        if baseline_mean_spacing > 0
                        else 0.0
                    )

                    spread_improvement = (
                        (baseline_mean_spread - ca_mean_spread)
                        / baseline_mean_spread
                        * 100
                        if baseline_mean_spread > 0
                        else 0.0
                    )

                    # ------------------------------------------
                    # Best Solutions
                    # ------------------------------------------

                    best_makespan = min(
                        best_pareto,
                        key=lambda c: c.makespan,
                    )

                    best_cost = min(
                        best_pareto,
                        key=lambda c: c.total_cost,
                    )

                    best_carbon = min(
                        best_pareto,
                        key=lambda c: c.total_carbon,
                    )

                    best_energy = min(
                        best_pareto,
                        key=lambda c: c.total_energy,
                    )
                    
                    # ---------------------------------------
                    # Diversity of priority lists
                    # ---------------------------------------

                    priority_set = set()

                    for c in best_pareto:

                        priority_set.add(
                            tuple(c.priority_list)
                        )


                    # ---------------------------------------
                    # Diversity of mode assignments
                    # ---------------------------------------

                    mode_set = set()

                    for c in best_pareto:

                        mode_set.add(
                            tuple(sorted(c.mode_assignment.items()))
                        )

                    writer.writerow(
                        [
                            benchmark.name,
                            len(project.activities),
                            population_size,
                            generations,
                            len(best_pareto),
                            best_makespan.makespan,
                            best_cost.total_cost,
                            best_carbon.total_carbon,
                            best_energy.total_energy,
                            ca_mean_hv,
                            ca_std_hv,
                            baseline_mean_hv,
                            baseline_std_hv,
                            hv_improvement,
                            ca_mean_igd,
                            ca_std_igd,
                            baseline_mean_igd,
                            baseline_std_igd,
                            igd_improvement,
                            ca_mean_spacing,
                            ca_std_spacing,
                            baseline_mean_spacing,
                            baseline_std_spacing,
                            spacing_improvement,
                            ca_mean_spread,
                            ca_std_spread,
                            baseline_mean_spread,
                            baseline_std_spread,
                            spread_improvement,
                            mean_time,
                            std_time,
                            "OK",
                        ]
                    )

                except Exception as exc:

                    import traceback
                    
                    raise

                    writer.writerow(
                        [
                            benchmark.name,
                            "",
                            population_size,
                            generations,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "FAILED",
                        ]
                    )
