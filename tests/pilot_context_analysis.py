"""
pilot_context_analysis.py

Extracts the real context trajectory of CA-NSGA-II
for the J30 pilot experiment.

For each instance and seed, the script records:
    - six context variables
    - adaptive crossover probability
    - adaptive mutation probability
    - best objective values
    - generation number

The analysis is based directly on NSGA2.history.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.models.project import Project
from src.parser.mm_parser import MMParser
from src.optimization.ca_nsga2 import NSGA2


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("benchmarks/data")

RESULT_ROOT = Path(
    "results/context_analysis_j30"
)

RAW_ROOT = RESULT_ROOT / "raw"

INSTANCES = [
    DATA_DIR / f"j3010_{i}.mm"
    for i in range(1, 11)
]

SEEDS = list(
    range(42, 52)
)

POPULATION_SIZE = 20
GENERATIONS = 20


# ============================================================
# Utility
# ============================================================

def clean_output() -> None:
    """
    Remove previous context-analysis output.
    """

    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_instances() -> None:
    """
    Verify that all requested PSPLIB instances exist.
    """

    print()
    print("=" * 80)
    print("INSTANCE VALIDATION")
    print("=" * 80)

    for instance in INSTANCES:

        if not instance.exists():

            raise FileNotFoundError(
                f"Missing instance: {instance}"
            )

        print(
            "Found:",
            instance.name,
        )

    print()
    print(
        f"Instances validated: {len(INSTANCES)}"
    )


# ============================================================
# One experiment
# ============================================================

def run_one(
    instance: Path,
    seed: int,
) -> list[dict]:

    print(
        f"Running {instance.name} | seed={seed}"
    )

    project = MMParser(
        instance
    ).parse()

    algorithm = NSGA2(
        project=project,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seed=seed,
        context_adaptive=True,
    )

    algorithm.run()

    history = algorithm.history

    if len(history) != GENERATIONS + 1:

        raise RuntimeError(
            f"Unexpected history length for "
            f"{instance.name}, seed={seed}: "
            f"{len(history)}"
        )

    records = []

    for item in history:

        record = {
            "instance": instance.name,
            "seed": seed,

            "generation":
                int(item["generation"]),

            "carbon_pressure":
                float(item["carbon_pressure"]),

            "energy_pressure":
                float(item["energy_pressure"]),

            "resource_pressure":
                float(item["resource_pressure"]),

            "cost_pressure":
                float(item["cost_pressure"]),

            "schedule_pressure":
                float(item["schedule_pressure"]),

            "uncertainty":
                float(item["uncertainty"]),

            "crossover_probability":
                float(item["crossover_probability"]),

            "mutation_probability":
                float(item["mutation_probability"]),

            "best_makespan":
                float(item["best_makespan"]),

            "best_cost":
                float(item["best_cost"]),

            "best_carbon":
                float(item["best_carbon"]),

            "best_energy":
                float(item["best_energy"]),
        }

        records.append(record)

    return records


# ============================================================
# Save one trajectory
# ============================================================

def save_trajectory(
    instance: Path,
    seed: int,
    records: list[dict],
) -> Path:

    directory = (
        RAW_ROOT
        / instance.stem
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / f"seed_{seed}.json"
    )

    payload = {
        "instance": instance.name,
        "seed": seed,
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "history_length": len(records),
        "history": records,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ============================================================
# Aggregate CSV
# ============================================================

def save_csv(
    records: list[dict],
) -> Path:

    import csv

    path = (
        RESULT_ROOT
        / "context_trajectory.csv"
    )

    if not records:
        raise RuntimeError(
            "No records available."
        )

    fieldnames = list(
        records[0].keys()
    )

    with path.open(
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

    return path


# ============================================================
# Validation
# ============================================================

def validate_records(
    records: list[dict],
) -> None:

    expected_runs = (
        len(INSTANCES)
        * len(SEEDS)
    )

    expected_records = (
        expected_runs
        * (GENERATIONS + 1)
    )

    print()
    print("=" * 80)
    print("TRAJECTORY VALIDATION")
    print("=" * 80)

    print(
        "Expected runs:",
        expected_runs,
    )

    print(
        "Actual runs:",
        len(
            set(
                (
                    r["instance"],
                    r["seed"],
                )
                for r in records
            )
        ),
    )

    print(
        "Expected records:",
        expected_records,
    )

    print(
        "Actual records:",
        len(records),
    )

    assert (
        len(records)
        == expected_records
    )

    assert (
        len(
            set(
                r["instance"]
                for r in records
            )
        )
        == len(INSTANCES)
    )

    assert (
        len(
            set(
                r["seed"]
                for r in records
            )
        )
        == len(SEEDS)
    )

    assert all(
        0 <= r["carbon_pressure"] <= 1
        for r in records
    )

    assert all(
        0 <= r["energy_pressure"] <= 1
        for r in records
    )

    assert all(
        0 <= r["resource_pressure"] <= 1
        for r in records
    )

    assert all(
        0 <= r["cost_pressure"] <= 1
        for r in records
    )

    assert all(
        0 <= r["schedule_pressure"] <= 1
        for r in records
    )

    assert all(
        0 <= r["uncertainty"] <= 1
        for r in records
    )

    assert all(
        0.0 <= r["crossover_probability"] <= 1.0
        for r in records
    )

    assert all(
        0.0 <= r["mutation_probability"] <= 1.0
        for r in records
    )

    print(
        "Range validation: PASS"
    )

    for instance in INSTANCES:

        for seed in SEEDS:

            subset = [
                r
                for r in records
                if (
                    r["instance"] == instance.name
                    and r["seed"] == seed
                )
            ]

            generations = [
                r["generation"]
                for r in subset
            ]

            assert generations == list(
                range(
                    GENERATIONS + 1
                )
            )

    print(
        "Generation continuity: PASS"
    )


# ============================================================
# Summary
# ============================================================

def create_summary(
    records: list[dict],
) -> dict:

    context_fields = [
        "carbon_pressure",
        "energy_pressure",
        "resource_pressure",
        "cost_pressure",
        "schedule_pressure",
        "uncertainty",
    ]

    operator_fields = [
        "crossover_probability",
        "mutation_probability",
    ]

    summary = {
        "instances": len(INSTANCES),
        "seeds": len(SEEDS),
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "total_runs":
            len(INSTANCES) * len(SEEDS),
        "records": len(records),
        "context_variables":
            context_fields,
        "operator_variables":
            operator_fields,
    }

    return summary


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CA-NSGA-II CONTEXT TRAJECTORY ANALYSIS")
    print("=" * 80)

    print(
        "Instances:",
        len(INSTANCES),
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Population:",
        POPULATION_SIZE,
    )

    print(
        "Generations:",
        GENERATIONS,
    )

    clean_output()

    validate_instances()

    all_records = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING CA-NSGA-II")
    print("=" * 80)

    for instance in INSTANCES:

        for seed in SEEDS:

            records = run_one(
                instance,
                seed,
            )

            save_trajectory(
                instance,
                seed,
                records,
            )

            all_records.extend(
                records
            )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_records(
        all_records
    )

    # --------------------------------------------------------
    # Save aggregate CSV
    # --------------------------------------------------------

    csv_path = save_csv(
        all_records
    )

    print()
    print(
        "Saved trajectory:",
        csv_path,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = create_summary(
        all_records
    )

    summary_path = (
        RESULT_ROOT
        / "context_analysis_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Saved summary:",
        summary_path,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CONTEXT TRAJECTORY ANALYSIS: COMPLETE")
    print("=" * 80)

    print(
        "Total runs:",
        summary["total_runs"],
    )

    print(
        "Total context records:",
        summary["records"],
    )


if __name__ == "__main__":
    main()