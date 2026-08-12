from __future__ import annotations

from dataclasses import dataclass, field

from src.models.scheduled_activity import ScheduledActivity


@dataclass(slots=True)
class DecodedSolution:
    """
    Output of SSGS / Decoder.

    Stores the generated schedule together with
    renewable-resource utilization information.
    """

    schedule: list[ScheduledActivity]

    makespan: float

    feasible: bool = True

    mode_assignment: dict[int, int] = field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Renewable-resource utilization
    # ---------------------------------------------------------

    resource_usage: list[list[float]] = field(
        default_factory=list
    )

    resource_capacities: list[float] = field(
        default_factory=list
    )
