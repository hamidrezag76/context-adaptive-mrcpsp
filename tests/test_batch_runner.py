from pathlib import Path
import shutil

from src.experiments.batch_runner import (
    BatchExperimentRunner,
)


def main():

    print(
        "\n========== BATCH RUNNER TEST =========="
    )

    instances = [
        Path("benchmarks/data/j3010_1.mm"),
        Path("benchmarks/data/j3010_2.mm"),
    ]

    seeds = [42, 43]

    result_root = Path(
        "results/test_batch"
    )

    if result_root.exists():
        shutil.rmtree(
            result_root
        )

    runner = BatchExperimentRunner(
        instances=instances,
        seeds=seeds,
        population_size=10,
        generations=5,
        result_root=result_root,
    )

    summaries = runner.run()

    # -------------------------------------------------
    # Instance validation
    # -------------------------------------------------

    assert set(
        summaries.keys()
    ) == {
        "j3010_1.mm",
        "j3010_2.mm",
    }

    assert len(
        summaries
    ) == 2

    print(
        "Instance execution: PASS"
    )

    # -------------------------------------------------
    # Runner validation
    # -------------------------------------------------

    for instance_name in (
        "j3010_1.mm",
        "j3010_2.mm",
    ):

        experiment = runner.get_runner(
            instance_name
        )

        assert (
            set(
                experiment.baseline_archives
            )
            == set(seeds)
        )

        assert (
            set(
                experiment.adaptive_archives
            )
            == set(seeds)
        )

        for seed in seeds:

            assert (
                len(
                    experiment
                    .baseline_archives[seed]
                )
                > 0
            )

            assert (
                len(
                    experiment
                    .adaptive_archives[seed]
                )
                > 0
            )

    print(
        "Seed validation: PASS"
    )

    # -------------------------------------------------
    # Metric validation
    # -------------------------------------------------

    for instance_name, summary in (
        summaries.items()
    ):

        assert (
            summary["hypervolume"]
            ["baseline_mean"]
            >= 0.0
        )

        assert (
            summary["hypervolume"]
            ["adaptive_mean"]
            >= 0.0
        )

        assert (
            summary["igd_plus"]
            ["baseline_mean"]
            >= 0.0
        )

        assert (
            summary["igd_plus"]
            ["adaptive_mean"]
            >= 0.0
        )

    print(
        "Metric validation: PASS"
    )

    # -------------------------------------------------
    # Persistence validation
    # -------------------------------------------------

    for instance in instances:

        for algorithm in (
            "baseline_nsga2",
            "ca_nsga2",
        ):

            for seed in seeds:

                path = (
                    result_root
                    / instance.stem
                    / algorithm
                    / f"seed_{seed}.json"
                )

                assert path.exists(), (
                    f"Missing result: {path}"
                )

    print(
        "Result persistence: PASS"
    )

    print(
        "\nBATCH RUNNER TEST: PASS"
    )


if __name__ == "__main__":
    main()
