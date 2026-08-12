from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy.stats import wilcoxon


ROOT = Path("results/pilot_j30")


def load_results():
    records = []

    for path in ROOT.rglob("*.json"):
        if path.name == "pilot_statistical_summary.json":
            continue

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        records.append(data)

    return records


def mean(values):
    return statistics.mean(values)


def ci95(values):
    n = len(values)

    if n < 2:
        return (float("nan"), float("nan"))

    m = mean(values)
    sd = statistics.stdev(values)

    # t critical value for df = 9
    t_critical = 2.262157

    margin = (
        t_critical
        * sd
        / math.sqrt(n)
    )

    return (
        m - margin,
        m + margin,
    )


def paired_effect_size(differences):
    """
    Cohen's dz for paired observations.
    """

    sd = statistics.stdev(
        differences
    )

    if sd == 0:
        return 0.0

    return (
        mean(differences)
        / sd
    )


def main():

    print()
    print("=" * 80)
    print("PILOT INSTANCE-LEVEL STATISTICS")
    print("=" * 80)

    records = load_results()

    print(
        f"JSON records: {len(records)}"
    )

    instances = sorted(
        set(
            r["instance"]
            for r in records
        )
    )

    print(
        f"Instances: {len(instances)}"
    )

    # -------------------------------------------------
    # Organize by instance
    # -------------------------------------------------

    grouped = {}

    for instance in instances:

        baseline = [
            r
            for r in records
            if (
                r["instance"] == instance
                and r["algorithm"]
                == "baseline_nsga2"
            )
        ]

        adaptive = [
            r
            for r in records
            if (
                r["instance"] == instance
                and r["algorithm"]
                == "ca_nsga2"
            )
        ]

        assert len(baseline) == 10
        assert len(adaptive) == 10

        baseline_by_seed = {
            r["seed"]: r
            for r in baseline
        }

        adaptive_by_seed = {
            r["seed"]: r
            for r in adaptive
        }

        assert (
            set(baseline_by_seed)
            == set(adaptive_by_seed)
        )

        grouped[instance] = (
            baseline_by_seed,
            adaptive_by_seed,
        )

    # -------------------------------------------------
    # Instance-level differences
    # -------------------------------------------------

    hv_differences = []
    igd_differences = []

    instance_rows = []

    print()
    print("=" * 80)
    print("INSTANCE-LEVEL RESULTS")
    print("=" * 80)

    for instance in instances:

        baseline, adaptive = (
            grouped[instance]
        )

        baseline_hv = mean(
            [
                r["metrics"]["hypervolume"]
                for r in baseline.values()
            ]
        )

        adaptive_hv = mean(
            [
                r["metrics"]["hypervolume"]
                for r in adaptive.values()
            ]
        )

        baseline_igd = mean(
            [
                r["metrics"]["igd_plus"]
                for r in baseline.values()
            ]
        )

        adaptive_igd = mean(
            [
                r["metrics"]["igd_plus"]
                for r in adaptive.values()
            ]
        )

        hv_diff = (
            adaptive_hv
            - baseline_hv
        )

        # Lower IGD+ is better.
        igd_diff = (
            baseline_igd
            - adaptive_igd
        )

        hv_improvement = (
            hv_diff
            / baseline_hv
            * 100.0
        )

        igd_improvement = (
            igd_diff
            / baseline_igd
            * 100.0
        )

        hv_differences.append(
            hv_diff
        )

        igd_differences.append(
            igd_diff
        )

        instance_rows.append(
            (
                instance,
                baseline_hv,
                adaptive_hv,
                hv_improvement,
                baseline_igd,
                adaptive_igd,
                igd_improvement,
            )
        )

        print()
        print(instance)

        print(
            f"  HV    : "
            f"Baseline={baseline_hv:.6f}, "
            f"CA={adaptive_hv:.6f}, "
            f"Improvement={hv_improvement:+.2f}%"
        )

        print(
            f"  IGD+  : "
            f"Baseline={baseline_igd:.6f}, "
            f"CA={adaptive_igd:.6f}, "
            f"Improvement={igd_improvement:+.2f}%"
        )

    # -------------------------------------------------
    # Overall instance-level statistics
    # -------------------------------------------------

    print()
    print("=" * 80)
    print("INSTANCE-LEVEL HYPERVOLUME")
    print("=" * 80)

    hv_mean = mean(
        hv_differences
    )

    hv_median = statistics.median(
        hv_differences
    )

    hv_sd = statistics.stdev(
        hv_differences
    )

    hv_ci = ci95(
        hv_differences
    )

    hv_wins = sum(
        d > 0
        for d in hv_differences
    )

    hv_losses = sum(
        d < 0
        for d in hv_differences
    )

    hv_ties = sum(
        d == 0
        for d in hv_differences
    )

    print(
        f"N: {len(hv_differences)}"
    )

    print(
        f"Mean difference: {hv_mean:.8f}"
    )

    print(
        f"Median difference: {hv_median:.8f}"
    )

    print(
        f"Std: {hv_sd:.8f}"
    )

    print(
        f"95% CI: "
        f"[{hv_ci[0]:.8f}, "
        f"{hv_ci[1]:.8f}]"
    )

    print(
        f"CA better: {hv_wins}"
    )

    print(
        f"Baseline better: {hv_losses}"
    )

    print(
        f"Ties: {hv_ties}"
    )

    print(
        f"Cohen's dz: "
        f"{paired_effect_size(hv_differences):.8f}"
    )

    # -------------------------------------------------
    # HV Wilcoxon
    # -------------------------------------------------

    if any(
        d != 0
        for d in hv_differences
    ):

        hv_w = wilcoxon(
            hv_differences,
            alternative="two-sided",
            zero_method="wilcox",
        )

        print()
        print(
            "Wilcoxon signed-rank test:"
        )

        print(
            f"  statistic = "
            f"{hv_w.statistic:.8f}"
        )

        print(
            f"  p-value   = "
            f"{hv_w.pvalue:.8f}"
        )

    # -------------------------------------------------
    # IGD+
    # -------------------------------------------------

    print()
    print("=" * 80)
    print("INSTANCE-LEVEL IGD+")
    print("=" * 80)

    igd_mean = mean(
        igd_differences
    )

    igd_median = statistics.median(
        igd_differences
    )

    igd_sd = statistics.stdev(
        igd_differences
    )

    igd_ci = ci95(
        igd_differences
    )

    igd_wins = sum(
        d > 0
        for d in igd_differences
    )

    igd_losses = sum(
        d < 0
        for d in igd_differences
    )

    igd_ties = sum(
        d == 0
        for d in igd_differences
    )

    print(
        f"N: {len(igd_differences)}"
    )

    print(
        f"Mean difference: {igd_mean:.8f}"
    )

    print(
        f"Median difference: {igd_median:.8f}"
    )

    print(
        f"Std: {igd_sd:.8f}"
    )

    print(
        f"95% CI: "
        f"[{igd_ci[0]:.8f}, "
        f"{igd_ci[1]:.8f}]"
    )

    print(
        f"CA better: {igd_wins}"
    )

    print(
        f"Baseline better: {igd_losses}"
    )

    print(
        f"Ties: {igd_ties}"
    )

    print(
        f"Cohen's dz: "
        f"{paired_effect_size(igd_differences):.8f}"
    )

    # -------------------------------------------------
    # IGD+ Wilcoxon
    # -------------------------------------------------

    if any(
        d != 0
        for d in igd_differences
    ):

        igd_w = wilcoxon(
            igd_differences,
            alternative="two-sided",
            zero_method="wilcox",
        )

        print()
        print(
            "Wilcoxon signed-rank test:"
        )

        print(
            f"  statistic = "
            f"{igd_w.statistic:.8f}"
        )

        print(
            f"  p-value   = "
            f"{igd_w.pvalue:.8f}"
        )

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    output = {
        "analysis": (
            "instance_level_paired_statistics"
        ),
        "n_instances": len(instances),
        "instances": instances,
        "hypervolume": {
            "mean_difference": hv_mean,
            "median_difference": hv_median,
            "std": hv_sd,
            "ci95": hv_ci,
            "ca_wins": hv_wins,
            "baseline_wins": hv_losses,
            "ties": hv_ties,
            "cohens_dz": paired_effect_size(
                hv_differences
            ),
            "wilcoxon": {
                "statistic": hv_w.statistic,
                "p_value": hv_w.pvalue,
            },
        },
        "igd_plus": {
            "mean_difference": igd_mean,
            "median_difference": igd_median,
            "std": igd_sd,
            "ci95": igd_ci,
            "ca_wins": igd_wins,
            "baseline_wins": igd_losses,
            "ties": igd_ties,
            "cohens_dz": paired_effect_size(
                igd_differences
            ),
            "wilcoxon": {
                "statistic": igd_w.statistic,
                "p_value": igd_w.pvalue,
            },
        },
        "instance_results": [
            {
                "instance": row[0],
                "baseline_hv": row[1],
                "adaptive_hv": row[2],
                "hv_improvement_percent": row[3],
                "baseline_igd_plus": row[4],
                "adaptive_igd_plus": row[5],
                "igd_plus_improvement_percent": row[6],
            }
            for row in instance_rows
        ],
    }

    output_path = (
        ROOT
        / "pilot_instance_level_statistics.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "INSTANCE-LEVEL STATISTICAL ANALYSIS: COMPLETE"
    )


if __name__ == "__main__":
    main()