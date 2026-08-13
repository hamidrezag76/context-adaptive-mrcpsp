from __future__ import annotations

from pathlib import Path
import shutil

from src.experiments.experimental_runner import (
    ExperimentalRunner,
)


def main():

    print(
        "\n========== EXPERIMENTAL RUNNER TEST =========="
    )

    result_root = Path(
        "results/test_runner"
    )

    if result_root.exists():
        shutil.rmtree(
            result_root
        )

    runner = ExperimentalRunner(
        instance=Path(
            "benchmarks/data/j3010_1.mm"
        ),
        seeds=[42, 43],
        population_size=10,
        generations=5,
        result_root=result_root,
    )

    summary = runner.run()

    # -------------------------------------------------
    # Basic validation
    # -------------------------------------------------

    assert len(
        runner.baseline_archives
    ) == 2

    assert len(
        runner.adaptive_archives
    ) == 2

    assert set(
        runner.baseline_archives
    ) == {42, 43}

    assert set(
        runner.adaptive_archives
    ) == {42, 43}

    # -------------------------------------------------
    # Metric validation
    # -------------------------------------------------

    assert (
        summary[
            "hypervolume"
        ][
            "baseline_mean"
        ] >= 0.0
    )

    assert (
        summary[
            "hypervolume"
        ][
            "adaptive_mean"
        ] >= 0.0
    )

    assert (
        summary[
            "igd_plus"
        ][
            "baseline_mean"
        ] >= 0.0
    )

    assert (
        summary[
            "igd_plus"
        ][
            "adaptive_mean"
        ] >= 0.0
    )

    # -------------------------------------------------
    # Stored files
    # -------------------------------------------------

    baseline_42 = (
        result_root
        / "j3010_1"
        / "baseline_nsga2"
        / "seed_42.json"
    )

    adaptive_42 = (
        result_root
        / "j3010_1"
        / "ca_nsga2"
        / "seed_42.json"
    )

    baseline_43 = (
        result_root
        / "j3010_1"
        / "baseline_nsga2"
        / "seed_43.json"
    )

    adaptive_43 = (
        result_root
        / "j3010_1"
        / "ca_nsga2"
        / "seed_43.json"
    )

    for path in [
        baseline_42,
        adaptive_42,
        baseline_43,
        adaptive_43,
    ]:

        assert path.exists(), (
            f"Missing result file: {path}"
        )

    # -------------------------------------------------
    # Verify stored metrics
    # -------------------------------------------------

    store = runner.result_store

    records = store.find(
        instance="j3010_1.mm"
    )

    assert len(records) == 6

    algorithms = {
        record["algorithm"]
        for record in records
    }

    assert algorithms == {
        "baseline_nsga2",
        "context_only_nsga2",
        "ca_nsga2",
    }

    for record in records:

        assert "metrics" in record

        assert (
            "hypervolume"
            in record["metrics"]
        )

        assert (
            "igd_plus"
            in record["metrics"]
        )

        assert record["metrics"]["hypervolume"] >= 0.0

        assert record["metrics"]["igd_plus"] >= 0.0

    for record in records:

        assert (
            "metrics"
            in record
        )

        assert (
            "hypervolume"
            in record["metrics"]
        )

        assert (
            "igd_plus"
            in record["metrics"]
        )

        assert (
            record["metrics"]["hypervolume"]
            >= 0.0
        )

        assert (
            record["metrics"]["igd_plus"]
            >= 0.0
        )

    print(
        "Runner execution: PASS"
    )

    print(
        "Common metrics: PASS"
    )

    print(
        "Result persistence: PASS"
    )

    print(
        "\nEXPERIMENTAL RUNNER TEST: PASS"
    )


if __name__ == "__main__":
    main()
