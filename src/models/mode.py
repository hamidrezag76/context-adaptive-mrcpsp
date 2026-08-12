from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Mode:
    """
    Execution mode of one activity.

    PSPLIB stores

        duration
        renewable resource demand
        nonrenewable resource demand

    Sustainable objectives

        cost
        carbon
        energy

    are generated later by the Context Engine.
    """

    id: int

    duration: int

    renewable: list[int] = field(default_factory=list)

    nonrenewable: list[int] = field(default_factory=list)

    cost: float = 0.0

    carbon: float = 0.0

    energy: float = 0.0
