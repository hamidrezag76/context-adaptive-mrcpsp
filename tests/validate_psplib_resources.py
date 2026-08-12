from pathlib import Path

from src.parser.mm_parser import MMParser


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks" / "data"


def parse_raw_resource_data(path: Path):
    """
    Extract resource structure, capacities, and mode resource
    demands directly from the raw PSPLIB file.
    """

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    # ---------------------------------------------------------
    # Resource counts
    # ---------------------------------------------------------

    renewable = None
    nonrenewable = None
    doubly = None

    for line in lines:

        upper = line.upper()

        if "RENEWABLE" in upper and "NONRENEWABLE" not in upper:
            numbers = [
                int(x)
                for x in line.replace(":", " ").split()
                if x.isdigit()
            ]

            if numbers:
                renewable = numbers[0]

        elif "NONRENEWABLE" in upper:
            numbers = [
                int(x)
                for x in line.replace(":", " ").split()
                if x.isdigit()
            ]

            if numbers:
                nonrenewable = numbers[0]

        elif "DOUBLY CONSTRAINED" in upper:
            numbers = [
                int(x)
                for x in line.replace(":", " ").split()
                if x.isdigit()
            ]

            if numbers:
                doubly = numbers[0]

    if renewable is None:
        raise ValueError("Renewable resource count not found.")

    if nonrenewable is None:
        raise ValueError(
            "Nonrenewable resource count not found."
        )

    if doubly is None:
        raise ValueError(
            "Doubly constrained resource count not found."
        )

    # ---------------------------------------------------------
    # Resource capacities
    # ---------------------------------------------------------

    availability_index = next(
        i
        for i, line in enumerate(lines)
        if "RESOURCEAVAILABILITIES" in line.upper()
    )

    capacities = None

    for line in lines[availability_index + 1:]:

        tokens = line.split()

        if not tokens:
            continue

        try:

            values = [
                int(x)
                for x in tokens
            ]

        except ValueError:

            continue

        expected = (
            renewable
            + nonrenewable
            + doubly
        )

        if len(values) >= expected:

            capacities = values[:expected]
            break

    if capacities is None:
        raise ValueError(
            "Resource capacities not found."
        )

    # ---------------------------------------------------------
    # Raw mode resource demands
    # ---------------------------------------------------------

    requests_index = next(
        i
        for i, line in enumerate(lines)
        if "REQUESTS/DURATIONS" in line.upper()
    )

    precedence_index = next(
        (
            i
            for i, line in enumerate(lines)
            if i > requests_index
            and "RESOURCEAVAILABILITIES" in line.upper()
        ),
        len(lines),
    )

    mode_data = {}

    current_activity = None

    for line in lines[
        requests_index + 1:precedence_index
    ]:

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("jobnr."):
            continue

        if stripped.startswith("-"):
            continue

        tokens = stripped.split()

        if len(tokens) < 2:
            continue

        # -----------------------------------------------------
        # New activity row
        # -----------------------------------------------------

        if line[:3].strip().isdigit():

            activity_id = int(tokens[0])

            mode_id = int(tokens[1])

            duration = int(tokens[2])

            resources = [
                int(x)
                for x in tokens[3:]
            ]

            current_activity = activity_id

        # -----------------------------------------------------
        # Continuation row
        # -----------------------------------------------------

        else:

            if current_activity is None:
                continue

            mode_id = int(tokens[0])

            duration = int(tokens[1])

            resources = [
                int(x)
                for x in tokens[2:]
            ]

        mode_data[
            (current_activity, mode_id)
        ] = (
            duration,
            tuple(resources),
        )

    return {
        "renewable": renewable,
        "nonrenewable": nonrenewable,
        "doubly": doubly,
        "capacities": tuple(capacities),
        "modes": mode_data,
    }


def validate_instance(path: Path):

    raw = parse_raw_resource_data(path)

    project = MMParser(path).parse()

    # ---------------------------------------------------------
    # Resource counts
    # ---------------------------------------------------------

    if project.renewable_count != raw["renewable"]:
        raise ValueError(
            "Renewable count mismatch: "
            f"raw={raw['renewable']}, "
            f"parsed={project.renewable_count}"
        )

    if project.nonrenewable_count != raw["nonrenewable"]:
        raise ValueError(
            "Nonrenewable count mismatch: "
            f"raw={raw['nonrenewable']}, "
            f"parsed={project.nonrenewable_count}"
        )

    if project.doubly_count != raw["doubly"]:
        raise ValueError(
            "Doubly constrained count mismatch: "
            f"raw={raw['doubly']}, "
            f"parsed={project.doubly_count}"
        )

    # ---------------------------------------------------------
    # Capacities
    # ---------------------------------------------------------

    parsed_capacities = tuple(
        project.renewable_capacities
        + project.nonrenewable_capacities
        + project.doubly_capacities
    )

    if parsed_capacities != raw["capacities"]:

        raise ValueError(
            "Resource capacity mismatch: "
            f"raw={raw['capacities']}, "
            f"parsed={parsed_capacities}"
        )

    # ---------------------------------------------------------
    # Resource objects
    # ---------------------------------------------------------

    expected_total = (
        raw["renewable"]
        + raw["nonrenewable"]
        + raw["doubly"]
    )

    if project.total_resources != expected_total:

        raise ValueError(
            "Total resource mismatch: "
            f"raw={expected_total}, "
            f"parsed={project.total_resources}"
        )

    # ---------------------------------------------------------
    # Mode resource demands
    # ---------------------------------------------------------

    parsed_modes = {}

    for activity in project.activities.values():

        for mode in activity.modes:

            demands = tuple(
                mode.renewable
                + mode.nonrenewable
            )

            parsed_modes[
                (activity.id, mode.id)
            ] = (
                mode.duration,
                demands,
            )

    raw_modes = raw["modes"]

    if set(parsed_modes) != set(raw_modes):

        missing = sorted(
            set(raw_modes) - set(parsed_modes)
        )

        unexpected = sorted(
            set(parsed_modes) - set(raw_modes)
        )

        raise ValueError(
            "Mode key mismatch: "
            f"missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    # ---------------------------------------------------------
    # Compare every mode
    # ---------------------------------------------------------

    for key in sorted(raw_modes):

        raw_duration, raw_resources = raw_modes[key]

        parsed_duration, parsed_resources = parsed_modes[key]

        if raw_duration != parsed_duration:

            raise ValueError(
                f"{key}: duration mismatch: "
                f"raw={raw_duration}, "
                f"parsed={parsed_duration}"
            )

        if tuple(raw_resources) != tuple(parsed_resources):

            raise ValueError(
                f"{key}: resource demand mismatch: "
                f"raw={raw_resources}, "
                f"parsed={parsed_resources}"
            )

        expected_length = (
            raw["renewable"]
            + raw["nonrenewable"]
            + raw["doubly"]
        )

        if len(parsed_resources) != expected_length:

            raise ValueError(
                f"{key}: invalid resource vector length: "
                f"expected={expected_length}, "
                f"actual={len(parsed_resources)}"
            )

    return {
        "activities": len(project.activities),
        "modes": project.total_modes,
        "resources": project.total_resources,
    }


def main():

    files = sorted(
        BENCHMARK_DIR.glob("*.mm")
    )

    print("=" * 70)
    print("PSPLIB RESOURCE VALIDATION — ALL INSTANCES")
    print("=" * 70)

    passed = 0
    failed = []

    for i, path in enumerate(files, start=1):

        try:

            result = validate_instance(path)

            passed += 1

            if (
                i % 50 == 0
                or i == len(files)
            ):

                print(
                    f"[{i}/{len(files)}] "
                    f"PASS — {path.name} "
                    f"(A={result['activities']}, "
                    f"M={result['modes']}, "
                    f"R={result['resources']})"
                )

        except Exception as exc:

            failed.append(
                (
                    path.name,
                    str(exc),
                )
            )

            print(
                f"[{i}/{len(files)}] "
                f"FAIL — {path.name} — {exc}"
            )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("Files:", len(files))
    print("Passed:", passed)
    print("Failed:", len(failed))

    if failed:

        print()
        print("FAILURES:")

        for name, error in failed:
            print(
                f"  {name}: {error}"
            )

        raise SystemExit(1)

    print()
    print("ALL RESOURCE VALIDATION TESTS PASSED.")


if __name__ == "__main__":
    main()
