from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.models.project import Project
from src.models.activity import Activity
from src.models.mode import Mode
from src.models.resource import Resource
from src.utils.constants import ResourceType
from src.context.sustainability_generator import SustainabilityGenerator


class MMParser:
    """
    ...docstring...
    """

    def __init__(self, filename: str | Path):

        self.filename = Path(filename)
        self.lines: List[str] = []
        self.index: int = 0

        self._load_file()

    # ======================================================

    def _load_file(self) -> None:

        with open(self.filename, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    # ======================================================

    def parse(self) -> Project:

        project = Project()

        self._read_project_information(project)

        self._read_resource_availability(project)

        self._read_precedence(project)

        self._read_modes(project)

        self._build_predecessors(project)
        
        SustainabilityGenerator(project).generate()

        project.reference_cost = sum(
            max(mode.cost for mode in activity.modes)
            for activity in project.activities.values()
        )

        project.reference_carbon = sum(
            max(mode.carbon for mode in activity.modes)
            for activity in project.activities.values()
        )

        project.reference_energy = sum(
            max(mode.energy for mode in activity.modes)
            for activity in project.activities.values()
        )

        project.baseline_cost = project.reference_cost
        project.baseline_carbon = project.reference_carbon
        project.baseline_energy = project.reference_energy

        return project

    # ======================================================

    def _next(self) -> str:

        line = self.lines[self.index]

        self.index += 1

        return line

    # ======================================================

    def _find_section(self, title: str):

        title = title.upper()

        for i, line in enumerate(self.lines):

            if title in line.upper():

                self.index = i + 1

                return

        raise ValueError(f"Section '{title}' not found.")
        # ======================================================

    def _read_project_information(self, project: Project):

        self.index = 0

        while self.index < len(self.lines):

            line = self._next().strip()

            if line == "":
                continue

            upper = line.upper()

            if "PRECEDENCE RELATIONS" in upper:
                self.index -= 1
                break

            if "NONRENEWABLE" in upper:
                numbers = [
                    int(x)
                    for x in line.replace(":", " ").split()
                    if x.isdigit()
                ]

                if numbers:
                    project.nonrenewable_count = numbers[0]

                continue


            if "DOUBLY CONSTRAINED" in upper:
                numbers = [
                    int(x)
                    for x in line.replace(":", " ").split()
                    if x.isdigit()
                ]

                if numbers:
                    project.doubly_count = numbers[0]

                continue


            if "RENEWABLE" in upper:
                numbers = [
                    int(x)
                    for x in line.replace(":", " ").split()
                    if x.isdigit()
                ]

                if numbers:
                    project.renewable_count = numbers[0]

                continue


            if "JOBS" in upper:
                numbers = [
                    int(x)
                    for x in line.replace(":", " ").split()
                    if x.isdigit()
                ]

                if numbers:
                    project.jobs = numbers[0]

                continue

            if "HORIZON" in upper:
                numbers = [int(x) for x in line.replace(":", " ").split() if x.isdigit()]
                if numbers:
                    project.horizon = numbers[0]
                continue

        # ======================================================

    def _read_precedence(self, project: Project):

        self._find_section("PRECEDENCE RELATIONS")

        # عبور از هدر جدول
        while self.index < len(self.lines):

            line = self._next().strip()

            if line.startswith("jobnr."):
                break

        activities: Dict[int, Activity] = {}

        while self.index < len(self.lines):

            line = self._next().strip()

            if line == "":
                continue

            upper = line.upper()

            if "REQUESTS/DURATIONS" in upper:
                self.index -= 1
                break

            if "REQUESTS" in upper and "DURATION" in upper:
                self.index -= 1
                break

            tokens = line.split()

            if len(tokens) < 3:
                continue
            
            if tokens[0].startswith("R"):
                break

            if tokens[0].startswith("N"):
                break

            if tokens[0].startswith("D"):
                break

            if not tokens[0].isdigit():
                continue

            activity_id = int(tokens[0])

            number_of_modes = int(tokens[1])

            successor_count = int(tokens[2])

            successors = []

            if successor_count > 0:

                successors = list(
                    map(int, tokens[3:3 + successor_count])
                )

            activity = Activity(

                id=activity_id,

                name=f"Activity {activity_id}",

                predecessors=set(),

                successors=set(successors),

                modes=[],

            )

            activities[activity_id] = activity

        project.activities = dict(activities)
        # ======================================================

    def _build_predecessors(self, project: Project):

        lookup = project.activities

        for activity in project.activities.values():

            for successor in activity.successors:

                if successor in lookup:

                    lookup[successor].predecessors.add(
                        activity.id
                    )
        # ======================================================

    def _read_modes(self, project: Project):
        """
        Read the REQUESTS/DURATIONS section of a PSPLIB .mm file.

        PSPLIB format:

            jobnr mode duration R... N...

        The job number appears only on the first mode of each
        activity. Continuation rows contain whitespace in the
        job-number column.
        """

        self._find_section("REQUESTS/DURATIONS")

        # ---------------------------------------------------------
        # Skip header
        # ---------------------------------------------------------

        while self.index < len(self.lines):

            line = self._next()

            if line.strip() == "":
                continue

            if line.strip().startswith("jobnr."):
                continue

            if line.strip().startswith("-"):
                break

        lookup = {
            activity.id: activity
            for activity in project.activities.values()
        }

        r = project.renewable_count
        n = project.nonrenewable_count
        d = project.doubly_count

        expected_resources = r + n + d

        current_activity: int | None = None

        # ---------------------------------------------------------
        # Read mode rows
        # ---------------------------------------------------------

        while self.index < len(self.lines):

            raw_line = self._next()

            if raw_line.strip() == "":
                continue

            if "RESOURCEAVAILABILITIES" in raw_line.upper():
                self.index -= 1
                break

            # -----------------------------------------------------
            # Determine whether job number is present.
            #
            # PSPLIB uses a fixed job-number column. Continuation
            # rows start with whitespace.
            # -----------------------------------------------------

            stripped = raw_line.strip()

            tokens = stripped.split()

            if len(tokens) < 2:
                continue

            # -----------------------------------------------------
            # New activity row
            #
            # Example:
            #
            # '  2      1     3       0    8    0    6'
            #
            # -----------------------------------------------------

            if raw_line[:3].strip().isdigit():

                activity_id = int(tokens[0])

                if activity_id not in lookup:
                    raise ValueError(
                        f"Unknown activity {activity_id} "
                        f"in REQUESTS/DURATIONS."
                    )

                if len(tokens) < 3:
                    raise ValueError(
                        f"Malformed mode row for activity "
                        f"{activity_id}: {raw_line!r}"
                    )

                mode_id = int(tokens[1])

                duration = int(tokens[2])

                resources = [
                    int(x)
                    for x in tokens[3:]
                ]

                current_activity = activity_id

            # -----------------------------------------------------
            # Continuation row
            #
            # Example:
            #
            # '         2     6       0    8    0    1'
            #
            # -----------------------------------------------------

            else:

                if current_activity is None:
                    raise ValueError(
                        "Found continuation mode before "
                        "any activity row."
                    )

                if len(tokens) < 2:
                    continue

                mode_id = int(tokens[0])

                duration = int(tokens[1])

                resources = [
                    int(x)
                    for x in tokens[2:]
                ]

            # -----------------------------------------------------
            # Validate resource vector
            # -----------------------------------------------------

            if len(resources) != expected_resources:

                raise ValueError(
                    f"Invalid resource vector for "
                    f"activity {current_activity}, "
                    f"mode {mode_id}: "
                    f"expected {expected_resources} values, "
                    f"got {len(resources)}. "
                    f"Raw line: {raw_line!r}"
                )

            # -----------------------------------------------------
            # Split resource requirements
            # -----------------------------------------------------

            renewable = resources[:r]

            nonrenewable = resources[
                r:r + n
            ]

            doubly = resources[
                r + n:r + n + d
            ]

            # -----------------------------------------------------
            # Create mode
            # -----------------------------------------------------

            mode = Mode(
                id=mode_id,
                duration=duration,
                renewable=renewable,
                nonrenewable=nonrenewable + doubly,
                cost=0.0,
                carbon=0.0,
                energy=0.0,
            )

            lookup[
                current_activity
            ].modes.append(mode)

    def _read_resource_availability(
        self,
        project: Project,
    ):

        self._find_section("RESOURCEAVAILABILITIES")

        capacities = []

        while self.index < len(self.lines):

            line = self._next().strip()

            if line == "":
                continue

            try:

                capacities = [

                    int(x)

                    for x in line.split()

                ]

                if capacities:
                    break

            except ValueError:

                continue

        r = project.renewable_count
        n = project.nonrenewable_count
        d = project.doubly_count

        project.renewable_capacities = capacities[:r]

        project.nonrenewable_capacities = capacities[r:r+n]

        project.doubly_capacities = capacities[r+n:r+n+d]

        # ---------------------------------------
        # Renewable
        # ---------------------------------------

        for i in range(r):

            project.add_renewable_resource(

                Resource(

                    id=i + 1,

                    name=f"R{i+1}",

                    resource_type=ResourceType.RENEWABLE,

                    capacity=project.renewable_capacities[i],

                )

            )

        # ---------------------------------------
        # Nonrenewable
        # ---------------------------------------

        for i in range(n):

            project.add_nonrenewable_resource(

                Resource(

                    id=r + i + 1,

                    name=f"N{i+1}",

                    resource_type=ResourceType.NON_RENEWABLE,

                    capacity=project.nonrenewable_capacities[i],

                )

            )

        # ---------------------------------------
        # Doubly constrained
        # ---------------------------------------

        for i in range(d):

            project.add_doubly_resource(
                Resource(
                    id=r + n + i + 1,
                    name=f"D{i+1}",
                    resource_type=ResourceType.DOUBLY_CONSTRAINED,
                    capacity=project.doubly_capacities[i],
                )
            )
