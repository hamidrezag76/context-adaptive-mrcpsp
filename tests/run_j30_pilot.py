from pathlib import Path

from src.experiments.batch_runner import (
    BatchExperimentRunner,
)


def main():

    print(
        "\n========== PSPLIB J30 PILOT =========="
    )

    data_dir = Path(
        "benchmarks/data"
    )

    instances = [
        data_dir / f"j3010_{i}.mm"
        for i in range(1, 11)
    ]

    seeds = list(
        range(42, 52)
    )

    population_size = 20
    generations = 20

    result_root = Path(
        "results/pilot_j30"
    )

    # -------------------------------------------------
    # Validate instances before starting
    # -------------------------------------------------

    print(
        "\n========== INSTANCE VALIDATION =========="
    )

    for instance in instances:

        if not instance.exists():

            raise FileNotFoundError(
                f"Missing instance: {instance}"
            )

        print(
            "Found:",
            instance.name,
        )

    print(
        "Instance validation: PASS"
    )

    # -------------------------------------------------
    # Run batch
    # -------------------------------------------------

    runner = BatchExperimentRunner(
        instances=instances,
        seeds=seeds,
        population_size=population_size,
        generations=generations,
        result_root=result_root,
    )

    summaries = runner.run()

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print(
        "\n========== PILOT SUMMARY =========="
    )

    for instance_name, summary in (
        summaries.items()
    ):

        hv = summary[
            "hypervolume"
        ]

        igd = summary[
            "igd_plus"
        ]

        print(
            f"\n{instance_name}"
        )

        print(
            "  HV baseline:",
            hv["baseline_mean"],
        )

        print(
            "  HV CA:",
            hv["adaptive_mean"],
        )

        print(
            "  HV improvement:",
            hv[
                "relative_improvement_percent"
            ],
        )

        print(
            "  IGD+ baseline:",
            igd["baseline_mean"],
        )

        print(
            "  IGD+ CA:",
            igd["adaptive_mean"],
        )

        print(
            "  IGD+ improvement:",
            igd[
                "relative_improvement_percent"
            ],
        )

    print(
        "\nPSPLIB J30 PILOT: COMPLETE"
    )


if __name__ == "__main__":
    main()
