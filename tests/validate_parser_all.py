from pathlib import Path
from src.parser.mm_parser import MMParser


def expected_mode_counts(path):
    lines = path.read_text(encoding="utf-8").splitlines()

    start = next(
        i for i, line in enumerate(lines)
        if "PRECEDENCE RELATIONS" in line.upper()
    )

    end = next(
        i for i, line in enumerate(lines[start:], start=start)
        if "REQUESTS/DURATIONS" in line.upper()
    )

    counts = {}

    for line in lines[start:end]:
        tokens = line.split()

        if len(tokens) < 3:
            continue

        if not tokens[0].isdigit():
            continue

        activity_id = int(tokens[0])
        mode_count = int(tokens[1])

        counts[activity_id] = mode_count

    return counts


files = sorted(
    Path("benchmarks/data").glob("*.mm")
)

print("=" * 70)
print("PARSER VALIDATION — ALL PSPLIB MM INSTANCES")
print("=" * 70)

passed = 0
failed = []

for i, path in enumerate(files, start=1):

    try:

        project = MMParser(path).parse()

        expected = expected_mode_counts(path)

        actual = {
            activity.id: activity.number_of_modes
            for activity in project.activities.values()
        }

        if len(project.activities) != len(expected):
            raise ValueError(
                f"Activity count mismatch: "
                f"parsed={len(project.activities)}, "
                f"expected={len(expected)}"
            )

        if expected != actual:
            mismatches = []

            for activity_id in sorted(
                set(expected) | set(actual)
            ):
                e = expected.get(activity_id)
                a = actual.get(activity_id)

                if e != a:
                    mismatches.append(
                        f"A{activity_id}: expected={e}, actual={a}"
                    )

            raise ValueError(
                "Mode-count mismatch: "
                + "; ".join(mismatches[:10])
            )

        passed += 1

        if i % 50 == 0 or i == len(files):
            print(
                f"[{i}/{len(files)}] "
                f"PASS — {path.name}"
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
print("ALL 640 INSTANCES PASSED.")
