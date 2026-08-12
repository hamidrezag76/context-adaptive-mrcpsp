from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ScheduledActivity:

    activity_id: int

    mode_id: int

    start: int

    finish: int