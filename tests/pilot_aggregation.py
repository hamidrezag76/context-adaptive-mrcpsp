import json
import statistics
from pathlib import Path
from collections import defaultdict


ROOT = Path("results/pilot_j30")


def main():

    data = defaultdict(
        lambda: {
            "baseline": [],
            "ca": [],
        }
    )

    files = list(ROOT.rglob("*.json"))

    for path in files:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            record = json.load(f)

        instance = record["instance"]

        algorithm = record["algorithm"]

        hv = float(
            record["metrics"]["hypervolume"]
        )

        igd = float(
            record["metrics"]["igd_plus"]
        )

        if algorithm == "baseline_nsga2":

            data[instance]["baseline"].append(
                (hv, igd)
            )

        elif algorithm == "ca_nsga2":

            data[instance]["ca"].append(
                (hv, igd)
            )

        else:

            raise ValueError(
                f"Unknown algorithm: {algorithm}"
            )

    print()
    print("=" * 80)
    print("PILOT J30 AGGREGATION")
    print("=" * 80)

    print(
        f"JSON files: {len(files)}"
    )

    print(
        f"Instances: {len(data)}"
    )

    print()

    for instance in sorted(data):

        baseline = data[instance]["baseline"]
        ca = data[instance]["ca"]

        if len(baseline) != 10:
            raise ValueError(
                f"{instance}: "
                f"baseline has {len(baseline)} seeds"
            )

        if len(ca) != 10:
            raise ValueError(
                f"{instance}: "
                f"CA has {len(ca)} seeds"
            )

        baseline_hv = [
            x[0]
            for x in baseline
        ]

        ca_hv = [
            x[0]
            for x in ca
        ]

        baseline_igd = [
            x[1]
            for x in baseline
        ]

        ca_igd = [
            x[1]
            for x in ca
        ]

        mean_baseline_hv = statistics.mean(
            baseline_hv
        )

        mean_ca_hv = statistics.mean(
            ca_hv
        )

        mean_baseline_igd = statistics.mean(
            baseline_igd
        )

        mean_ca_igd = statistics.mean(
            ca_igd
        )

        hv_improvement = (
            (
                mean_ca_hv
                - mean_baseline_hv
            )
            / mean_baseline_hv
            * 100.0
        )

        igd_improvement = (
            (
                mean_baseline_igd
                - mean_ca_igd
            )
            / mean_baseline_igd
            * 100.0
        )

        hv_ca_better = sum(
            ca_hv[i] > baseline_hv[i]
            for i in range(10)
        )

        igd_ca_better = sum(
            ca_igd[i] < baseline_igd[i]
            for i in range(10)
        )

        print(
            f"{instance}"
        )

        print(
            f"  HV   : "
            f"Baseline={mean_baseline_hv:.6f}  "
            f"CA={mean_ca_hv:.6f}  "
            f"Improvement={hv_improvement:+.2f}%"
        )

        print(
            f"  IGD+ : "
            f"Baseline={mean_baseline_igd:.6f}  "
            f"CA={mean_ca_igd:.6f}  "
            f"Improvement={igd_improvement:+.2f}%"
        )

        print(
            f"  CA better: "
            f"HV={hv_ca_better}/10  "
            f"IGD+={igd_ca_better}/10"
        )

        print()


if __name__ == "__main__":

    main()