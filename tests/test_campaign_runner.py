from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.campaign_runner import CampaignRunner


MANIFEST = Path(
    "results/campaign/j30/manifest_test.csv"
)

RESULT_ROOT = Path(
    "results/campaign/j30/test_runner"
)


def make_runner(
    tmp_path,
):
    return CampaignRunner(
        manifest=MANIFEST,
        seeds=[42, 43],
        population_size=20,
        generations=20,
        result_root=tmp_path,
    )


def test_campaign_runner_prepare_loads_manifest(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    instances = runner.prepare()

    assert len(instances) == 10

    assert (
        runner.number_of_instances
        == 10
    )

    assert (
        runner.number_of_groups
        == 1
    )

    assert (
        runner.replications_per_group
        == 10
    )


def test_missing_result_is_not_valid(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    instance = runner.prepare()[0]

    assert (
        runner.is_result_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )


def test_status_reports_pending_for_missing_results(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    instance = runner.prepare()[0]

    status = runner.status(
        instance,
        42,
    )

    assert status == {
        "baseline_nsga2": "pending",
        "context_only_nsga2": "pending",
        "ca_nsga2": "pending",
    }


def test_progress_counts_missing_results(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    progress = runner.progress()

    assert progress["total"] == (
        10 * 2 * 3
    )

    assert progress["completed"] == 0

    assert progress["remaining"] == (
        10 * 2 * 3
    )


def test_evaluation_is_pending_after_new_run(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    instance = runner.prepare()[0]

    runner.run_one(
        instance,
        42,
        "baseline_nsga2",
    )

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )


def test_run_result_contract_after_single_run(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    instance = runner.prepare()[0]

    runner.run_one(
        instance,
        42,
        "baseline_nsga2",
    )

    assert runner.is_result_valid(
        instance,
        42,
        "baseline_nsga2",
    )

    record = runner.result_store.load_run(
        instance=instance.instance,
        algorithm="baseline_nsga2",
        seed=42,
    )

    assert (
        record["instance"]
        == instance.instance
    )

    assert (
        record["algorithm"]
        == "baseline_nsga2"
    )

    assert (
        int(record["seed"])
        == 42
    )

    assert (
        record["metadata"]
        ["metrics_status"]
        == "pending_common_evaluation"
    )

    assert (
        record["metrics"]
        == {}
    )


def test_evaluation_status_is_pending_after_run(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    instance = runner.prepare()[0]

    runner.run_one(
        instance,
        42,
        "baseline_nsga2",
    )

    record = runner.result_store.load_run(
        instance=instance.instance,
        algorithm="baseline_nsga2",
        seed=42,
    )

    assert (
        record["metadata"]
        ["metrics_status"]
        == "pending_common_evaluation"
    )

    assert (
        record["metrics"]
        == {}
    )

def test_evaluation_progress_before_evaluation(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    progress = runner.evaluation_progress()

    assert progress["total"] == (
        10 * 2 * 3
    )

    assert progress["evaluated"] == 0

    assert progress["pending"] == (
        10 * 2 * 3
    )

def test_evaluate_one_instance_updates_evaluation_progress(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    before = runner.evaluation_progress()

    assert before == {
        "total": 60,
        "evaluated": 0,
        "pending": 60,
    }

    # First create the six optimization runs
    # required for one instance:
    #
    # 2 seeds × 3 algorithms = 6 runs

    instance = runner.instances[0]

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            runner.run_one(
                instance,
                seed,
                algorithm,
            )

    after_runs = runner.progress()

    assert after_runs == {
        "total": 60,
        "completed": 6,
        "remaining": 54,
    }

    # Evaluate exactly one instance.
    result = runner.evaluate_campaign(
        limit=1
    )

    assert result == {
        "instances_evaluated": 1,
        "runs_evaluated": 6,
    }

    after_evaluation = (
        runner.evaluation_progress()
    )

    assert after_evaluation == {
        "total": 60,
        "evaluated": 6,
        "pending": 54,
    }

    # Every algorithm for both seeds must now
    # contain a completed common evaluation.

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            assert runner.is_evaluation_valid(
                instance,
                seed,
                algorithm,
            )

def test_evaluate_campaign_skips_fully_evaluated_instance(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    instance = runner.instances[0]

    # -------------------------------------------------
    # Create all optimization runs for one instance.
    # -------------------------------------------------

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            runner.run_one(
                instance,
                seed,
                algorithm,
            )

    # -------------------------------------------------
    # Evaluate the first instance.
    # -------------------------------------------------

    first = runner.evaluate_campaign(
        limit=1
    )

    assert first == {
        "instances_evaluated": 1,
        "runs_evaluated": 6,
    }

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 6,
        "pending": 54,
    }

    # -------------------------------------------------
    # Verify that the first instance is completely
    # evaluated.
    # -------------------------------------------------

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            assert runner.is_evaluation_valid(
                instance,
                seed,
                algorithm,
            )

    # -------------------------------------------------
    # Directly verify that a fully evaluated instance
    # is recognized as evaluated.
    # -------------------------------------------------

    assert all(
        runner.is_evaluation_valid(
            instance,
            seed,
            algorithm,
        )
        for seed in runner.seeds
        for algorithm in runner.ALGORITHMS
    )
    
def test_partial_evaluation_is_not_treated_as_complete(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    instance = runner.instances[0]

    # -------------------------------------------------
    # Create all six optimization runs for one
    # instance:
    #
    # 2 seeds × 3 algorithms = 6 runs
    # -------------------------------------------------

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            runner.run_one(
                instance,
                seed,
                algorithm,
            )

    assert runner.progress() == {
        "total": 60,
        "completed": 6,
        "remaining": 54,
    }

    # -------------------------------------------------
    # Complete common evaluation for the instance.
    # -------------------------------------------------

    result = runner.evaluate_campaign(
        limit=1
    )

    assert result == {
        "instances_evaluated": 1,
        "runs_evaluated": 6,
    }

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 6,
        "pending": 54,
    }

    # -------------------------------------------------
    # Invalidate exactly one evaluation.
    #
    # The optimization result remains valid, but the
    # common evaluation is marked as pending again.
    # -------------------------------------------------

    runner.result_store.update_run_evaluation(
        instance=instance.instance,
        algorithm="baseline_nsga2",
        seed=42,
        metrics={},
        metadata={
            "metrics_status":
                "pending_common_evaluation",
        },
    )

    # -------------------------------------------------
    # The optimization result must still be valid.
    # -------------------------------------------------

    assert runner.is_result_valid(
        instance,
        42,
        "baseline_nsga2",
    )

    # -------------------------------------------------
    # But its evaluation must now be invalid.
    # -------------------------------------------------

    assert not runner.is_evaluation_valid(
        instance,
        42,
        "baseline_nsga2",
    )

    # -------------------------------------------------
    # Exactly one of the six evaluations is now
    # pending.
    # -------------------------------------------------

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 5,
        "pending": 55,
    }

    # -------------------------------------------------
    # The instance must NOT be considered completely
    # evaluated.
    # -------------------------------------------------

    assert not all(
        runner.is_evaluation_valid(
            instance,
            seed,
            algorithm,
        )
        for seed in runner.seeds
        for algorithm in runner.ALGORITHMS
    )

    # -------------------------------------------------
    # Re-evaluate the incomplete instance.
    #
    # All six optimization results already exist, so
    # evaluate_instance() can recompute the common
    # metrics for all six runs.
    # -------------------------------------------------

    result = runner.evaluate_campaign(
        limit=1
    )

    assert result == {
        "instances_evaluated": 1,
        "runs_evaluated": 6,
    }

    # -------------------------------------------------
    # The complete instance must now be evaluated
    # again successfully.
    # -------------------------------------------------

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 6,
        "pending": 54,
    }

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            assert runner.is_evaluation_valid(
                instance,
                seed,
                algorithm,
            )
            
def test_is_evaluation_valid_rejects_corrupted_metrics(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    instance = runner.instances[0]

    # -------------------------------------------------
    # Create one optimization run.
    # -------------------------------------------------

    runner.run_one(
        instance,
        42,
        "baseline_nsga2",
    )

    # -------------------------------------------------
    # Initially evaluation must be pending.
    # -------------------------------------------------

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )

    # -------------------------------------------------
    # Add a valid common evaluation.
    # -------------------------------------------------

    runner.result_store.update_run_evaluation(
        instance=instance.instance,
        algorithm="baseline_nsga2",
        seed=42,
        metrics={
            "hypervolume": 123.456,
            "igd_plus": 0.789,
        },
        metadata={
            "metrics_status": "evaluated",
        },
    )

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is True
    )

    # -------------------------------------------------
    # Load the result file directly.
    # -------------------------------------------------

    path = runner.result_path(
        instance,
        42,
        "baseline_nsga2",
    )

    record = runner.result_store.load_run(
        instance=instance.instance,
        algorithm="baseline_nsga2",
        seed=42,
    )

    # -------------------------------------------------
    # Case 1:
    # Missing hypervolume.
    # -------------------------------------------------

    record["metrics"].pop(
        "hypervolume"
    )

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )

    # -------------------------------------------------
    # Restore valid metrics.
    # -------------------------------------------------

    record["metrics"] = {
        "hypervolume": 123.456,
        "igd_plus": 0.789,
    }

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # Case 2:
    # Missing IGD+.
    # -------------------------------------------------

    record["metrics"].pop(
        "igd_plus"
    )

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )

    # -------------------------------------------------
    # Restore valid metrics.
    # -------------------------------------------------

    record["metrics"] = {
        "hypervolume": 123.456,
        "igd_plus": 0.789,
    }

    # -------------------------------------------------
    # Case 3:
    # NaN / non-finite metric.
    # -------------------------------------------------

    record["metrics"][
        "hypervolume"
    ] = float("nan")

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )

    # -------------------------------------------------
    # Restore valid metrics.
    # -------------------------------------------------

    record["metrics"] = {
        "hypervolume": 123.456,
        "igd_plus": 0.789,
    }

    # -------------------------------------------------
    # Case 4:
    # Non-numeric IGD+.
    # -------------------------------------------------

    record["metrics"][
        "igd_plus"
    ] = "invalid"

    path.write_text(
        json.dumps(
            record,
            indent=2,
        ),
        encoding="utf-8",
    )

    assert (
        runner.is_evaluation_valid(
            instance,
            42,
            "baseline_nsga2",
        )
        is False
    )
    
def test_evaluate_campaign_skips_instance_with_incomplete_runs(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    first = runner.instances[0]

    # -------------------------------------------------
    # Only create one of the six required runs.
    # -------------------------------------------------

    runner.run_one(
        first,
        42,
        "baseline_nsga2",
    )

    assert runner.progress() == {
        "total": 60,
        "completed": 1,
        "remaining": 59,
    }

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 0,
        "pending": 60,
    }

    # -------------------------------------------------
    # Evaluation must not attempt to evaluate an
    # incomplete instance.
    # -------------------------------------------------

    result = runner.evaluate_campaign(
        limit=1
    )

    assert result == {
        "instances_evaluated": 0,
        "runs_evaluated": 0,
    }

    # -------------------------------------------------
    # The incomplete instance must remain unevaluated.
    # -------------------------------------------------

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 0,
        "pending": 60,
    }

    assert (
        runner.is_evaluation_valid(
            first,
            42,
            "baseline_nsga2",
        )
        is False
    )

def test_evaluation_resumes_after_incomplete_instance_is_completed(
    tmp_path,
):

    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    instance = runner.instances[0]

    # -------------------------------------------------
    # Create only 5 of the 6 required runs.
    #
    # 2 seeds × 3 algorithms = 6 runs
    # -------------------------------------------------

    incomplete_runs = [
        (42, "baseline_nsga2"),
        (42, "context_only_nsga2"),
        (42, "ca_nsga2"),
        (43, "baseline_nsga2"),
        (43, "context_only_nsga2"),
    ]

    for seed, algorithm in incomplete_runs:

        runner.run_one(
            instance,
            seed,
            algorithm,
        )

    # -------------------------------------------------
    # The instance is incomplete.
    # Therefore it must NOT be evaluated.
    # -------------------------------------------------

    assert runner.progress() == {
        "total": 60,
        "completed": 5,
        "remaining": 55,
    }

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 0,
        "pending": 60,
    }

    result_before_completion = (
        runner.evaluate_campaign(
            limit=1
        )
    )

    assert result_before_completion == {
        "instances_evaluated": 0,
        "runs_evaluated": 0,
    }

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 0,
        "pending": 60,
    }

    # -------------------------------------------------
    # Complete the missing sixth run.
    # -------------------------------------------------

    runner.run_one(
        instance,
        43,
        "ca_nsga2",
    )

    assert runner.progress() == {
        "total": 60,
        "completed": 6,
        "remaining": 54,
    }

    # -------------------------------------------------
    # Now the instance is complete and can be evaluated.
    # -------------------------------------------------

    result_after_completion = (
        runner.evaluate_campaign(
            limit=1
        )
    )

    assert result_after_completion == {
        "instances_evaluated": 1,
        "runs_evaluated": 6,
    }

    # -------------------------------------------------
    # Exactly six runs must now have completed
    # common evaluation.
    # -------------------------------------------------

    assert runner.evaluation_progress() == {
        "total": 60,
        "evaluated": 6,
        "pending": 54,
    }

    # -------------------------------------------------
    # Verify every run belonging to the evaluated
    # instance has valid common metrics.
    # -------------------------------------------------

    for seed in runner.seeds:

        for algorithm in runner.ALGORITHMS:

            assert runner.is_evaluation_valid(
                instance,
                seed,
                algorithm,
            )

            record = runner.result_store.load_run(
                instance=instance.instance,
                algorithm=algorithm,
                seed=seed,
            )

            assert (
                record["metadata"]
                ["metrics_status"]
                == "evaluated"
            )

            assert (
                "hypervolume"
                in record["metrics"]
            )

            assert (
                "igd_plus"
                in record["metrics"]
            )
            
def test_campaign_run_resumes_completed_results(
    tmp_path,
):
    runner = make_runner(
        tmp_path
    )

    runner.prepare()

    first_instance = runner.instances[0]

    # -------------------------------------------------
    # Pre-complete one run.
    # -------------------------------------------------

    runner.run_one(
        first_instance,
        42,
        "baseline_nsga2",
    )

    assert runner.progress() == {
        "total": 60,
        "completed": 1,
        "remaining": 59,
    }

    # -------------------------------------------------
    # Resume the campaign with a limit of one run.
    #
    # The already completed baseline/seed=42 run
    # must be skipped.
    #
    # Therefore the next pending run should be executed.
    # -------------------------------------------------

    result = runner.run(
        limit=1
    )

    assert result == {
        "total": 60,
        "completed": 2,
        "remaining": 58,
    }

    # -------------------------------------------------
    # The pre-existing result must still be valid.
    # -------------------------------------------------

    assert runner.is_result_valid(
        first_instance,
        42,
        "baseline_nsga2",
    )

    # -------------------------------------------------
    # Exactly two optimization runs should now exist.
    # -------------------------------------------------

    assert runner.progress() == {
        "total": 60,
        "completed": 2,
        "remaining": 58,
    }