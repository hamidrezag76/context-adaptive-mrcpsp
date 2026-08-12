"""
schedule.py

Production Ready Schedule Model

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ScheduledActivity:
    """
    One scheduled activity.
    """

    id: int

    mode_id: int

    start_time: float

    finish_time: float

    duration: float

    renewable_usage: dict[int, float] = field(default_factory=dict)

    nonrenewable_usage: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class Schedule:
    """
    Complete project schedule.
    """

    activities: list[ScheduledActivity]

    makespan: float

    total_cost: float

    total_carbon: float

    total_energy: float

    feasible: bool = True

    penalty: float = 0.0

    def get_activity(
        self,
        activity_id: int,
    ) -> ScheduledActivity:

        for activity in self.activities:

            if activity.id == activity_id:

                return activity

        raise KeyError(
            f"Activity {activity_id} not found."
        )

    @property
    def activity_count(
        self,
    ) -> int:

        return len(self.activities)

    def start_times(
        self,
    ) -> dict[int, float]:

        return {
            activity.id: activity.start_time
            for activity in self.activities
        }

    def finish_times(
        self,
    ) -> dict[int, float]:

        return {
            activity.id: activity.finish_time
            for activity in self.activities
        }
