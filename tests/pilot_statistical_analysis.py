from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

from scipy.stats import ttest_rel, wilcoxon


ROOT = Path("results/pilot_j30")


def load_data():

    data = defaultdict(
        lambda: {
            "baseline": {},
            "ca": {},
        }
    )

    for path in ROOT.rglob("*.json"):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            record = json.load(f)

        instance = record["instance"]
        seed = int(record["seed"])
        algorithm = record["algorithm"]

        hv = float(
            record["metrics"]["hypervolume"]
        )

        igd = float(
            record["metrics"]["igd_plus"]
        )

        data[instance][
            "ca" if algorithm == "ca_nsga2"
            else "baseline"
        ][seed] = {
            "hv": hv,
            "igd": igd,
        }

    return data


def mean(values):

    return statistics.mean(values)


def median(values):

    return statistics.median(values)


def std(values):

    return statistics.stdev(values)


def confidence_interval(values):

    n = len(values)

    if n < 2:
        return (float("nan"), float("nan"))

    m = mean(values)
    s = std(values)

    se = s / math.sqrt(n)

    # 95% normal approximation.
    # For the pilot summary this is descriptive;
    # the paired Wilcoxon test is also reported.
    margin = 1.96 * se

    return (
        m - margin,
        m + margin,
    )


def analyze():

    data = load_data()

    hv_differences = []
    igd_differences = []

    paired_records = []

    for instance in sorted(data):

        baseline = data[instance]["baseline"]
        ca = data[instance]["ca"]

        assert set(baseline) == set(ca)

        for seed in sorted(baseline):

            baseline_hv = baseline[seed]["hv"]
            ca_hv = ca[seed]["hv"]

            baseline_igd = baseline[seed]["igd"]
            ca_igd = ca[seed]["igd"]

            hv_diff = (
                ca_hv
                - baseline_hv
            )

            # Positive means CA is better.
            igd_diff = (
                baseline_igd
                - ca_igd
            )

            hv_differences.append(
                hv_diff
            )

            igd_differences.append(
                igd_diff
            )

            paired_records.append(
                {
                    "instance": instance,
                    "seed": seed,
                    "hv_difference": hv_diff,
                    "igd_difference": igd_diff,
                }
            )

    return (
        hv_differences,
        igd_differences,
        paired_records,
    )


def print_analysis(
    name,
    differences,
):

    ci_low, ci_high = (
        confidence_interval(
            differences
        )
    )

    print()
    print(
        "=" * 80
    )

    print(name)

    print(
        "=" * 80
    )

    print(
        "N:",
        len(differences),
    )

    print(
        "Mean difference:",
        f"{mean(differences):.8f}",
    )

    print(
        "Median difference:",
        f"{median(differences):.8f}",
    )

    print(
        "Std:",
        f"{std(differences):.8f}",
    )

    print(
        "95% CI:",
        f"[{ci_low:.8f}, {ci_high:.8f}]",
    )

    positive = sum(
        x > 0
        for x in differences
    )

    negative = sum(
        x < 0
        for x in differences
    )

    ties = sum(
        x == 0
        for x in differences
    )

    print(
        "CA better:",
        positive,
    )

    print(
        "Baseline better:",
        negative,
    )

    print(
        "Ties:",
        ties,
    )

    return {
        "mean": mean(differences),
        "median": median(differences),
        "std": std(differences),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "positive": positive,
        "negative": negative,
        "ties": ties,
    }


def main():

    print()
    print(
        "=" * 80
    )

    print(
        "PILOT STATISTICAL ANALYSIS"
    )

    print(
        "=" * 80
    )

    hv, igd, records = analyze()

    print(
        f"Paired observations: {len(records)}"
    )

    hv_summary = print_analysis(
        "HYPERVOLUME",
        hv,
    )

    igd_summary = print_analysis(
        "IGD+",
        igd,
    )

    # -------------------------------------------------
    # Paired t-tests
    # -------------------------------------------------

    hv_t = ttest_rel(
        hv,
        [0.0] * len(hv),
    )

    igd_t = ttest_rel(
        igd,
        [0.0] * len(igd),
    )

    # -------------------------------------------------
    # Wilcoxon signed-rank tests
    # -------------------------------------------------

    hv_w = wilcoxon(
        hv,
        zero_method="wilcox",
        alternative="two-sided",
    )

    igd_w = wilcoxon(
        igd,
        zero_method="wilcox",
        alternative="two-sided",
    )

    print()
    print(
        "=" * 80
    )

    print(
        "PAIRED STATISTICAL TESTS"
    )

    print(
        "=" * 80
    )

    print(
        "HV paired t-test:"
    )

    print(
        f"  statistic = {hv_t.statistic:.8f}"
    )

    print(
        f"  p-value   = {hv_t.pvalue:.8f}"
    )

    print()

    print(
        "IGD+ paired t-test:"
    )

    print(
        f"  statistic = {igd_t.statistic:.8f}"
    )

    print(
        f"  p-value   = {igd_t.pvalue:.8f}"
    )

    print()

    print(
        "HV Wilcoxon:"
    )

    print(
        f"  statistic = {hv_w.statistic:.8f}"
    )

    print(
        f"  p-value   = {hv_w.pvalue:.8f}"
    )

    print()

    print(
        "IGD+ Wilcoxon:"
    )

    print(
        f"  statistic = {igd_w.statistic:.8f}"
    )

    print(
        f"  p-value   = {igd_w.pvalue:.8f}"
    )

    # -------------------------------------------------
    # Instance-level summary
    # -------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "INSTANCE-LEVEL WIN COUNTS"
    )

    print(
        "=" * 80
    )

    instance_summary = {}

    for instance in sorted(
        {
            r["instance"]
            for r in records
        }
    ):

        subset = [
            r
            for r in records
            if r["instance"] == instance
        ]

        hv_wins = sum(
            r["hv_difference"] > 0
            for r in subset
        )

        igd_wins = sum(
            r["igd_difference"] > 0
            for r in subset
        )

        instance_summary[instance] = {
            "hv_ca_wins": hv_wins,
            "igd_ca_wins": igd_wins,
        }

        print(
            f"{instance}: "
            f"HV={hv_wins}/10, "
            f"IGD+={igd_wins}/10"
        )

    # -------------------------------------------------
    # Save statistical summary
    # -------------------------------------------------

    output = {
        "observations": len(records),
        "hypervolume": hv_summary,
        "igd_plus": igd_summary,
        "paired_t_test": {
            "hypervolume": {
                "statistic": float(
                    hv_t.statistic
                ),
                "p_value": float(
                    hv_t.pvalue
                ),
            },
            "igd_plus": {
                "statistic": float(
                    igd_t.statistic
                ),
                "p_value": float(
                    igd_t.pvalue
                ),
            },
        },
        "wilcoxon": {
            "hypervolume": {
                "statistic": float(
                    hv_w.statistic
                ),
                "p_value": float(
                    hv_w.pvalue
                ),
            },
            "igd_plus": {
                "statistic": float(
                    igd_w.statistic
                ),
                "p_value": float(
                    igd_w.pvalue
                ),
            },
        },
        "instance_summary": instance_summary,
    }

    output_path = (
        ROOT
        / "pilot_statistical_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_path,
    )

    print()
    print(
        "PILOT STATISTICAL ANALYSIS: COMPLETE"
    )


if __name__ == "__main__":

    main()