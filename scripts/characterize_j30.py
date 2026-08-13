"""
characterize_j30.py

Structural characterization of the complete J30 PSPLIB benchmark set.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from src.parser.mm_parser import MMParser


ROOT = Path(".")
BENCHMARK_DIR = ROOT / "benchmarks" / "data"
OUTPUT = (
    ROOT
    / "results"
    / "campaign"
    / "j30"
    / "characterization.csv"
)


def parse_group_replication(
    filename: str,
) -> tuple[int, int]:

    stem = Path(filename).stem

    if not stem.startswith("j30"):
        raise ValueError(
            f"Invalid J30 filename: {filename}"
        )

    body = stem[3:]

    group_text, replication_text = body.split("_")

    return (
        int(group_text),
        int(replication_text),
    )


def safe_mean(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return statistics.fmean(values)


def safe_std(
    values: list[float],
) -> float:

    if len(values) < 2:
        return 0.0

    return statistics.stdev(values)


def coefficient_of_variation(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    mean = safe_mean(values)

    if mean == 0.0:
        return 0.0

    return safe_std(values) / mean


def characterize(
    path: Path,
) -> dict[str, object]:

    project = MMParser(path).parse()

    activities = project.activities_list

    # ---------------------------------------------------------
    # Basic project structure
    # ---------------------------------------------------------

    activity_count = (
        project.number_of_activities
    )

    total_modes = (
        project.total_modes
    )

    renewable_count = (
        project.number_of_renewable_resources
    )

    nonrenewable_count = (
        project.number_of_nonrenewable_resources
    )

    doubly_count = (
        project.number_of_doubly_constrained_resources
    )

    total_resources = (
        project.total_resources
    )

    precedence_relations = (
        project.total_edges
    )

    # ---------------------------------------------------------
    # Precedence topology
    # ---------------------------------------------------------

    successor_counts = [
        float(activity.outdegree)
        for activity in activities
    ]

    predecessor_counts = [
        float(activity.indegree)
        for activity in activities
    ]

    start_activity_count = sum(
        1
        for activity in activities
        if activity.is_start_activity
    )

    finish_activity_count = sum(
        1
        for activity in activities
        if activity.is_finish_activity
    )

    # ---------------------------------------------------------
    # Modes per activity
    # ---------------------------------------------------------

    modes_per_activity = [
        float(activity.number_of_modes)
        for activity in activities
    ]

    # ---------------------------------------------------------
    # Duration characteristics
    # ---------------------------------------------------------

    durations = []

    mode_duration_ranges = []

    mode_duration_cvs = []

    for activity in activities:

        activity_durations = [
            float(mode.duration)
            for mode in activity.modes
        ]

        durations.extend(
            activity_durations
        )

        if activity_durations:

            mode_duration_ranges.append(
                max(activity_durations)
                - min(activity_durations)
            )

            mean_duration = safe_mean(
                activity_durations
            )

            if mean_duration > 0.0:

                mode_duration_cvs.append(
                    safe_std(
                        activity_durations
                    )
                    / mean_duration
                )

    # ---------------------------------------------------------
    # Resource-demand characteristics
    # ---------------------------------------------------------

    renewable_demands = []

    nonrenewable_demands = []

    total_demands = []

    mode_resource_totals = []

    mode_resource_cvs = []

    for activity in activities:

        for mode in activity.modes:

            renewable = [
                float(value)
                for value in mode.renewable
            ]

            nonrenewable = [
                float(value)
                for value in mode.nonrenewable
            ]

            all_demands = (
                renewable
                + nonrenewable
            )

            renewable_demands.extend(
                renewable
            )

            nonrenewable_demands.extend(
                nonrenewable
            )

            total_demands.extend(
                all_demands
            )

            if all_demands:

                mode_total = sum(
                    all_demands
                )

                mode_resource_totals.append(
                    mode_total
                )

                mean_demand = safe_mean(
                    all_demands
                )

                if mean_demand > 0.0:

                    mode_resource_cvs.append(
                        safe_std(
                            all_demands
                        )
                        / mean_demand
                    )

    # ---------------------------------------------------------
    # Resource capacities
    # ---------------------------------------------------------

    capacities = [
        float(resource.capacity)
        for resource in project.resources
    ]

    # ---------------------------------------------------------
    # Sustainability-related mode characteristics
    # ---------------------------------------------------------

    mode_costs = []

    mode_carbon = []

    mode_energy = []

    for activity in activities:

        for mode in activity.modes:

            mode_costs.append(
                float(mode.cost)
            )

            mode_carbon.append(
                float(mode.carbon)
            )

            mode_energy.append(
                float(mode.energy)
            )

    group, replication = (
        parse_group_replication(
            path.name
        )
    )

    return {

        "instance":
            path.name,

        "group":
            group,

        "replication":
            replication,

        # -----------------------------------------------------
        # Basic structure
        # -----------------------------------------------------

        "activities":
            activity_count,

        "total_modes":
            total_modes,

        "mean_modes_per_activity":
            safe_mean(
                modes_per_activity
            ),

        "renewable_resources":
            renewable_count,

        "nonrenewable_resources":
            nonrenewable_count,

        "doubly_constrained_resources":
            doubly_count,

        "resources":
            total_resources,

        "precedence_relations":
            precedence_relations,

        # -----------------------------------------------------
        # Network topology
        # -----------------------------------------------------

        "start_activities":
            start_activity_count,

        "finish_activities":
            finish_activity_count,

        "mean_successors":
            safe_mean(
                successor_counts
            ),

        "std_successors":
            safe_std(
                successor_counts
            ),

        "max_successors":
            max(
                successor_counts,
                default=0.0,
            ),

        "mean_predecessors":
            safe_mean(
                predecessor_counts
            ),

        "std_predecessors":
            safe_std(
                predecessor_counts
            ),

        "max_predecessors":
            max(
                predecessor_counts,
                default=0.0,
            ),

        # -----------------------------------------------------
        # Duration
        # -----------------------------------------------------

        "min_duration":
            min(
                durations,
                default=0.0,
            ),

        "max_duration":
            max(
                durations,
                default=0.0,
            ),

        "mean_duration":
            safe_mean(
                durations
            ),

        "std_duration":
            safe_std(
                durations
            ),

        "duration_cv":
            coefficient_of_variation(
                durations
            ),

        "mean_mode_duration_range":
            safe_mean(
                mode_duration_ranges
            ),

        "mean_mode_duration_cv":
            safe_mean(
                mode_duration_cvs
            ),

        # -----------------------------------------------------
        # Resource demand
        # -----------------------------------------------------

        "min_resource_demand":
            min(
                total_demands,
                default=0.0,
            ),

        "max_resource_demand":
            max(
                total_demands,
                default=0.0,
            ),

        "mean_resource_demand":
            safe_mean(
                total_demands
            ),

        "std_resource_demand":
            safe_std(
                total_demands
            ),

        "resource_demand_cv":
            coefficient_of_variation(
                total_demands
            ),

        "mean_mode_resource_total":
            safe_mean(
                mode_resource_totals
            ),

        "std_mode_resource_total":
            safe_std(
                mode_resource_totals
            ),

        "mean_mode_resource_cv":
            safe_mean(
                mode_resource_cvs
            ),

        # -----------------------------------------------------
        # Capacity
        # -----------------------------------------------------

        "min_capacity":
            min(
                capacities,
                default=0.0,
            ),

        "max_capacity":
            max(
                capacities,
                default=0.0,
            ),

        "mean_capacity":
            safe_mean(
                capacities
            ),

        "std_capacity":
            safe_std(
                capacities
            ),

        # -----------------------------------------------------
        # Sustainability mode values
        # -----------------------------------------------------

        "mean_mode_cost":
            safe_mean(
                mode_costs
            ),

        "mean_mode_carbon":
            safe_mean(
                mode_carbon
            ),

        "mean_mode_energy":
            safe_mean(
                mode_energy
            ),

        "path":
            str(
                path
            ),
    }


def main() -> None:

    instances = sorted(
        BENCHMARK_DIR.glob("j30*.mm")
    )

    if not instances:

        raise FileNotFoundError(
            "No J30 benchmark instances found."
        )

    records = []

    for path in instances:

        records.append(
            characterize(path)
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        records[0].keys()
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    groups = {
        int(record["group"])
        for record in records
    }

    replications = {}

    for record in records:

        group = int(
            record["group"]
        )

        replications[group] = (
            replications.get(group, 0)
            + 1
        )

    print(
        "J30 characterization created successfully."
    )

    print(
        "Instances:",
        len(records),
    )

    print(
        "Groups:",
        len(groups),
    )

    print(
        "Replications per group:",
        min(replications.values()),
        "to",
        max(replications.values()),
    )

    print(
        "Features:",
        len(fieldnames),
    )

    print(
        "Output:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()