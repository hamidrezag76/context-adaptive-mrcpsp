from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from src.experiments.result_store import ResultStore


def main():

    print(
        "\n========== RESULT STORE TEST =========="
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="ca_smrcpsp_result_store_"
        )
    )

    try:

        store = ResultStore(
            temporary_directory
        )

        archive = [
            (
                34.0,
                849549.0946,
                82944.4033,
                112517.8188,
            ),
            (
                36.0,
                840000.0,
                83000.0,
                113000.0,
            ),
        ]

        metrics = {
            "hypervolume": 0.6547549,
            "igd_plus": 0.1627344,
        }

        best = (
            34.0,
            849549.0946,
            82944.4033,
            112517.8188,
        )

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        path = store.save_run(
            instance="j3010_1.mm",
            algorithm="ca_nsga2",
            seed=42,
            population_size=20,
            generations=20,
            archive_points=archive,
            metrics=metrics,
            best_objectives=best,
            metadata={
                "context_adaptive": True,
                "test": True,
            },
        )

        print(
            "Saved:",
            path,
        )

        assert path.exists()

        # -------------------------------------------------
        # Load
        # -------------------------------------------------

        record = store.load_run(
            instance="j3010_1.mm",
            algorithm="ca_nsga2",
            seed=42,
        )

        assert (
            record["instance"]
            == "j3010_1.mm"
        )

        assert (
            record["algorithm"]
            == "ca_nsga2"
        )

        assert (
            record["seed"]
            == 42
        )

        assert (
            record["population_size"]
            == 20
        )

        assert (
            record["generations"]
            == 20
        )

        assert (
            record["archive_size"]
            == 2
        )

        assert (
            len(
                record["archive_objectives"]
            )
            == 2
        )

        assert (
            record["metrics"]["hypervolume"]
            == metrics["hypervolume"]
        )

        print(
            "Save/load validation: PASS"
        )

        # -------------------------------------------------
        # Evaluation update
        # -------------------------------------------------

        updated_path = (
            store.update_run_evaluation(
                instance="j3010_1.mm",
                algorithm="ca_nsga2",
                seed=42,
                metrics={
                    "hypervolume": 0.8125,
                    "igd_plus": 0.0942,
                },
                metadata={
                    "metric_reference_set":
                        "common_all_three_modes",
                    "metric_reference_point":
                        [1.05, 1.05, 1.05, 1.05],
                    "reference_set_size": 8,
                    "metrics_status":
                        "evaluated",
                },
            )
        )

        assert updated_path.exists()

        updated_record = (
            store.load_run(
                instance="j3010_1.mm",
                algorithm="ca_nsga2",
                seed=42,
            )
        )

        assert (
            updated_record["metrics"]
            ["hypervolume"]
            == 0.8125
        )

        assert (
            updated_record["metrics"]
            ["igd_plus"]
            == 0.0942
        )

        assert (
            updated_record["metadata"]
            ["metrics_status"]
            == "evaluated"
        )

        assert (
            updated_record["metadata"]
            ["metric_reference_set"]
            == "common_all_three_modes"
        )

        assert (
            updated_record["archive_objectives"]
            == [
                list(point)
                for point in archive
            ]
        )

        assert (
            updated_record["best_objectives"]
            == list(best)
        )

        print(
            "Evaluation update: PASS"
        )

        # -------------------------------------------------
        # Find
        # -------------------------------------------------

        records = store.find(
            instance="j3010_1.mm",
            algorithm="ca_nsga2",
            seeds=[42],
        )

        assert len(records) == 1

        print(
            "Filtering validation: PASS"
        )

        # -------------------------------------------------
        # Duplicate protection
        # -------------------------------------------------

        duplicate_blocked = False

        try:

            store.save_run(
                instance="j3010_1.mm",
                algorithm="ca_nsga2",
                seed=42,
                population_size=20,
                generations=20,
                archive_points=archive,
                metrics=metrics,
                best_objectives=best,
            )

        except FileExistsError:

            duplicate_blocked = True

        assert duplicate_blocked

        print(
            "Duplicate protection: PASS"
        )

        # -------------------------------------------------
        # CSV
        # -------------------------------------------------

        csv_path = (
            temporary_directory
            / "summary.csv"
        )

        store.export_csv(
            csv_path
        )

        assert csv_path.exists()

        csv_content = csv_path.read_text(
            encoding="utf-8"
        )

        assert "hypervolume" in csv_content

        assert "igd_plus" in csv_content

        print(
            "CSV export: PASS"
        )

        # -------------------------------------------------
        # Final
        # -------------------------------------------------

        print(
            "\nRESULT STORE TEST: PASS"
        )

    finally:

        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
