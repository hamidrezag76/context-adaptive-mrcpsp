from __future__ import annotations

from typing import Dict, List

from src.models.project import Project
from src.models.activity import Activity
from src.models.mode import Mode
from src.optimization.decoded_solution import DecodedSolution
from src.models.scheduled_activity import ScheduledActivity


class SSGS:
    """
    Serial Schedule Generation Scheme.

    Generates a feasible schedule while explicitly
    preserving renewable-resource utilization data
    for downstream context analysis.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:

        self.project = project

    # ---------------------------------------------------------
    # Generate Schedule
    # ---------------------------------------------------------

    def generate(
        self,
        priority_list: List[int],
        mode_assignment: Dict[int, int],
    ) -> DecodedSolution:

        activities = self.project.activities

        renewable_capacity = list(
            self.project.renewable_capacities
        )

        num_resources = len(
            renewable_capacity
        )

        # Extra scheduling buffer retained from the
        # original implementation.
        horizon = self.project.horizon + 200

        # usage[t][r] = renewable resource r consumed
        # during time period t.
        usage = [
            [0.0] * num_resources
            for _ in range(horizon)
        ]

        finish_times: dict[int, float] = {}

        schedule: list[ScheduledActivity] = []

        missing = [
            i
            for i in priority_list
            if i not in activities
        ]

        if missing:

            raise ValueError(
                f"Activities not found: {missing}"
            )

        priority_index = {
            activity_id: i
            for i, activity_id in enumerate(
                priority_list
            )
        }

        scheduled = set()

        remaining = set(
            activities.keys()
        )

        # -----------------------------------------------------
        # Schedule all activities
        # -----------------------------------------------------

        while remaining:

            eligible = []

            for activity_id in remaining:

                activity = activities[
                    activity_id
                ]

                if all(
                    predecessor in scheduled
                    for predecessor in activity.predecessors
                ):

                    eligible.append(
                        activity_id
                    )

            if not eligible:

                raise RuntimeError(
                    "No eligible activity found."
                )

            # Respect chromosome priority list.
            eligible.sort(
                key=lambda x: priority_index[x]
            )

            activity_id = eligible[0]

            activity = activities[
                activity_id
            ]

            mode = self._get_mode(
                activity,
                mode_assignment[activity_id],
            )

            # -------------------------------------------------
            # Earliest start from precedence constraints
            # -------------------------------------------------

            earliest = 0

            if activity.predecessors:

                earliest = max(
                    finish_times[p]
                    for p in activity.predecessors
                )

            start = int(earliest)

            # -------------------------------------------------
            # Renewable-resource feasibility
            # -------------------------------------------------

            while start < horizon:

                if self._resource_feasible(
                    usage,
                    renewable_capacity,
                    mode,
                    start,
                ):
                    break

                start += 1

            if start >= horizon:

                raise RuntimeError(
                    f"Unable to schedule activity "
                    f"{activity_id} within horizon."
                )

            finish = (
                start
                + mode.duration
            )

            # -------------------------------------------------
            # Reserve renewable resources
            # -------------------------------------------------

            self._reserve(
                usage,
                mode,
                start,
            )

            finish_times[
                activity_id
            ] = finish

            schedule.append(
                ScheduledActivity(
                    activity_id=activity_id,
                    mode_id=mode.id,
                    start=start,
                    finish=finish,
                )
            )

            scheduled.add(
                activity_id
            )

            remaining.remove(
                activity_id
            )

        # -----------------------------------------------------
        # Makespan
        # -----------------------------------------------------

        makespan = max(
            activity.finish
            for activity in schedule
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Preserve only the actually relevant time horizon
        # for downstream context analysis.
        #
        # The original SSGS uses horizon + 200 as a safety
        # buffer. We do NOT want those unused trailing zeros
        # to artificially reduce resource utilization.
        # -----------------------------------------------------

        actual_horizon = int(
            makespan
        )

        actual_usage = [
            row.copy()
            for row in usage[
                :actual_horizon
            ]
        ]

        # -----------------------------------------------------
        # Return decoded solution
        # -----------------------------------------------------

        return DecodedSolution(
            schedule=schedule,
            makespan=makespan,
            feasible=True,
            resource_usage=actual_usage,
            resource_capacities=[
                float(c)
                for c in renewable_capacity
            ],
        )

    # ---------------------------------------------------------
    # Mode lookup
    # ---------------------------------------------------------

    def _get_mode(
        self,
        activity: Activity,
        mode_id: int,
    ) -> Mode:

        for mode in activity.modes:

            if mode.id == mode_id:

                return mode

        raise ValueError(
            f"Mode {mode_id} not found "
            f"for activity {activity.id}"
        )

    # ---------------------------------------------------------
    # Resource feasibility
    # ---------------------------------------------------------

    def _resource_feasible(
        self,
        usage,
        capacity,
        mode,
        start,
    ) -> bool:

        finish = (
            start
            + mode.duration
        )

        for t in range(
            start,
            finish,
        ):

            for r in range(
                len(mode.renewable)
            ):

                required = (
                    mode.renewable[r]
                )

                if (
                    usage[t][r]
                    + required
                    > capacity[r]
                ):

                    return False

        return True

    # ---------------------------------------------------------
    # Reserve resources
    # ---------------------------------------------------------

    def _reserve(
        self,
        usage,
        mode,
        start,
    ) -> None:

        finish = (
            start
            + mode.duration
        )

        for t in range(
            start,
            finish,
        ):

            for r in range(
                len(mode.renewable)
            ):

                usage[t][r] += (
                    mode.renewable[r]
                )
