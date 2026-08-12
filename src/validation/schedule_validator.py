from __future__ import annotations

from src.models.project import Project
from src.optimization.decoded_solution import DecodedSolution


class ScheduleValidator:
    """
    Independent feasibility validator for decoded schedules.

    Checks:
        1. All project activities are scheduled.
        2. Activity modes are valid.
        3. Start/finish times are consistent with durations.
        4. Precedence constraints are satisfied.
        5. Renewable-resource constraints are satisfied.
        6. Reported makespan is consistent with the schedule.
    """

    def __init__(self, project: Project) -> None:
        self.project = project

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def validate(
        self,
        decoded: DecodedSolution,
    ) -> None:
        """
        Validate a decoded solution.

        Raises
        ------
        ValueError
            If any feasibility condition is violated.
        """

        self._validate_activity_coverage(decoded)

        self._validate_modes(decoded)

        self._validate_timing(decoded)

        self._validate_precedence(decoded)

        self._validate_renewable_resources(decoded)

        self._validate_makespan(decoded)

    # ---------------------------------------------------------
    # Activity coverage
    # ---------------------------------------------------------

    def _validate_activity_coverage(
        self,
        decoded: DecodedSolution,
    ) -> None:

        scheduled_ids = {
            item.activity_id
            for item in decoded.schedule
        }

        expected_ids = set(
            self.project.activities.keys()
        )

        missing = expected_ids - scheduled_ids

        extra = scheduled_ids - expected_ids

        if missing:
            raise ValueError(
                f"Missing scheduled activities: "
                f"{sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unknown scheduled activities: "
                f"{sorted(extra)}"
            )

        if len(decoded.schedule) != len(expected_ids):
            raise ValueError(
                "Schedule contains duplicate activities."
            )

    # ---------------------------------------------------------
    # Mode validation
    # ---------------------------------------------------------

    def _validate_modes(
        self,
        decoded: DecodedSolution,
    ) -> None:

        for activity_id, mode_id in (
            decoded.mode_assignment.items()
        ):

            if activity_id not in self.project.activities:
                raise ValueError(
                    f"Unknown activity in mode assignment: "
                    f"{activity_id}"
                )

            activity = self.project.get_activity(
                activity_id
            )

            valid_modes = {
                mode.id
                for mode in activity.modes
            }

            if mode_id not in valid_modes:
                raise ValueError(
                    f"Invalid mode {mode_id} "
                    f"for activity {activity_id}."
                )

        expected_ids = set(
            self.project.activities.keys()
        )

        assigned_ids = set(
            decoded.mode_assignment.keys()
        )

        missing = expected_ids - assigned_ids

        if missing:
            raise ValueError(
                f"Missing mode assignments: "
                f"{sorted(missing)}"
            )

    # ---------------------------------------------------------
    # Timing
    # ---------------------------------------------------------

    def _validate_timing(
        self,
        decoded: DecodedSolution,
    ) -> None:

        for item in decoded.schedule:

            activity = self.project.get_activity(
                item.activity_id
            )

            mode = activity.get_mode(
                item.mode_id
            )

            if item.start < 0:
                raise ValueError(
                    f"Negative start time for "
                    f"activity {item.activity_id}."
                )

            if item.finish < item.start:
                raise ValueError(
                    f"Finish before start for "
                    f"activity {item.activity_id}."
                )

            expected_finish = (
                item.start + mode.duration
            )

            if item.finish != expected_finish:
                raise ValueError(
                    f"Timing mismatch for activity "
                    f"{item.activity_id}: "
                    f"expected finish={expected_finish}, "
                    f"actual finish={item.finish}"
                )

    # ---------------------------------------------------------
    # Precedence
    # ---------------------------------------------------------

    def _validate_precedence(
        self,
        decoded: DecodedSolution,
    ) -> None:

        schedule_by_id = {
            item.activity_id: item
            for item in decoded.schedule
        }

        for activity in (
            self.project.activities.values()
        ):

            current = schedule_by_id[
                activity.id
            ]

            for predecessor_id in (
                activity.predecessors
            ):

                predecessor = schedule_by_id[
                    predecessor_id
                ]

                if current.start < predecessor.finish:

                    raise ValueError(
                        f"Precedence violation: "
                        f"activity {predecessor_id} "
                        f"must finish before activity "
                        f"{activity.id} starts."
                    )

    # ---------------------------------------------------------
    # Renewable resources
    # ---------------------------------------------------------

    def _validate_renewable_resources(
        self,
        decoded: DecodedSolution,
    ) -> None:

        capacities = (
            self.project.renewable_capacities
        )

        if not capacities:
            return

        for t in range(
            0,
            int(decoded.makespan) + 1,
        ):

            usage = [
                0
                for _ in capacities
            ]

            for item in decoded.schedule:

                if not (
                    item.start <= t < item.finish
                ):
                    continue

                activity = (
                    self.project.get_activity(
                        item.activity_id
                    )
                )

                mode = activity.get_mode(
                    item.mode_id
                )

                for r, demand in enumerate(
                    mode.renewable
                ):

                    if r >= len(capacities):
                        raise ValueError(
                            f"Resource index {r} "
                            f"exceeds capacity vector."
                        )

                    usage[r] += demand

                    if usage[r] > capacities[r]:

                        raise ValueError(
                            f"Renewable resource "
                            f"violation at t={t}, "
                            f"resource={r + 1}: "
                            f"usage={usage[r]}, "
                            f"capacity={capacities[r]}"
                        )

    # ---------------------------------------------------------
    # Makespan
    # ---------------------------------------------------------

    def _validate_makespan(
        self,
        decoded: DecodedSolution,
    ) -> None:

        if not decoded.schedule:

            expected = 0.0

        else:

            expected = float(
                max(
                    item.finish
                    for item in decoded.schedule
                )
            )

        if float(decoded.makespan) != expected:

            raise ValueError(
                f"Makespan mismatch: "
                f"reported={decoded.makespan}, "
                f"expected={expected}"
            )
