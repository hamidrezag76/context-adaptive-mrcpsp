from __future__ import annotations

from pathlib import Path
import math
import shutil

from src.experiments.experimental_runner import ExperimentalRunner


def test_experimental_protocol():
    """
    Validates the complete multi-seed experimental protocol.

    This test verifies:

    1. Seed consistency
    2. 4D objective dimensionality
    3. Common normalization
    4. Common reference set
    5. Common reference point
    6. Per-seed metrics
    7. Summary metrics
    8. Result persistence
    """

    result_root = Path("results/test_protocol")

    if result_root.exists():
        shutil.rmtree(result_root)

    seeds = [42, 43, 44]

    runner = ExperimentalRunner(
        instance=Path(
            "benchmarks/data/j3010_1.mm"
        ),
        seeds=seeds,
        population_size=10,
        generations=5,
        result_root=result_root,
    )

    summary = runner.run()

    evaluator = runner.evaluate_metrics()

    # -------------------------------------------------
    # Seed consistency
    # -------------------------------------------------

    assert evaluator.seeds == seeds

    assert set(
        runner.baseline_archives
    ) == set(seeds)

    assert set(
        runner.adaptive_archives
    ) == set(seeds)

    # -------------------------------------------------
    # Objective dimensionality
    # -------------------------------------------------

    for seed in seeds:

        for point in (
            runner.baseline_archives[seed]
            + runner.adaptive_archives[seed]
        ):

            assert len(point) == 4

            assert all(
                math.isfinite(float(x))
                for x in point
            )

    # -------------------------------------------------
    # Common normalization
    # -------------------------------------------------

    assert evaluator.normalizer is not None

    for seed in seeds:

        for point in (
            evaluator.baseline_normalized[seed]
            + evaluator.adaptive_normalized[seed]
        ):

            assert len(point) == 4

            assert all(
                0.0 <= float(x) <= 1.0
                for x in point
            )

    # -------------------------------------------------
    # Common reference set
    # -------------------------------------------------

    assert len(
        evaluator.reference_set
    ) > 0

    assert len(
        evaluator.reference_set[0]
    ) == 4

    # -------------------------------------------------
    # Common reference point
    # -------------------------------------------------

    assert len(
        evaluator.reference_point
    ) == 4

    for value in evaluator.reference_point:

        assert math.isfinite(
            float(value)
        )

        assert value > 0.0

    # -------------------------------------------------
    # Per-seed metric validation
    # -------------------------------------------------

    results = evaluator.evaluate()

    assert len(results) == len(seeds)

    for result in results:

        assert result["seed"] in seeds

        assert (
            result["baseline_hv"]
            >= 0.0
        )

        assert (
            result["adaptive_hv"]
            >= 0.0
        )

        assert (
            result["baseline_igd_plus"]
            >= 0.0
        )

        assert (
            result["adaptive_igd_plus"]
            >= 0.0
        )

    # -------------------------------------------------
    # Summary validation
    # -------------------------------------------------

    assert (
        summary["hypervolume"]["baseline_mean"]
        >= 0.0
    )

    assert (
        summary["hypervolume"]["adaptive_mean"]
        >= 0.0
    )

    assert (
        summary["igd_plus"]["baseline_mean"]
        >= 0.0
    )

    assert (
        summary["igd_plus"]["adaptive_mean"]
        >= 0.0
    )

    # -------------------------------------------------
    # Result persistence
    # -------------------------------------------------

    records = runner.result_store.find(
        instance="j3010_1.mm",
        seeds=seeds,
    )

    assert len(records) == 6

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

        assert "metadata" in record

        assert (
            record["metadata"]
            ["metric_reference_set"]
            == "common_all_seeds"
        )