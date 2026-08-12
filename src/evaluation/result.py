from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationResult:
    """
    Multi-objective evaluation result.

    In addition to the four optimization objectives,
    the result stores resource-utilization information
    extracted from the decoded schedule.
    """

    makespan: float

    total_cost: float

    total_carbon: float

    total_energy: float

    feasible: bool = True

    penalty: float = 0.0

    resource_pressure: float = 0.0

    average_resource_utilization: float = 0.0

    peak_resource_utilization: float = 0.0
