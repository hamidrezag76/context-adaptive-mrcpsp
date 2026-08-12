"""
context.py

Context representation for the Context-Adaptive
Sustainable Multi-Mode RCPSP.

This module stores all environmental indicators
that influence adaptive optimization.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Context:
    """
    Dynamic project context.

    Every value is normalized to [0,1].

    These indicators are NOT objectives.

    They are only used to guide the search.
    """

    carbon_pressure: float

    energy_pressure: float

    resource_pressure: float

    cost_pressure: float

    schedule_pressure: float

    uncertainty: float

    # -----------------------------------------

    def as_vector(self):

        return [

            self.carbon_pressure,

            self.energy_pressure,

            self.resource_pressure,

            self.cost_pressure,

            self.schedule_pressure,

            self.uncertainty,

        ]

    # -----------------------------------------

    @property
    def size(self):

        return 6

    # -----------------------------------------

    def copy(self):

        return Context(

            carbon_pressure=self.carbon_pressure,

            energy_pressure=self.energy_pressure,

            resource_pressure=self.resource_pressure,

            cost_pressure=self.cost_pressure,

            schedule_pressure=self.schedule_pressure,

            uncertainty=self.uncertainty,

        )

    # -----------------------------------------

    def clip(self):

        self.carbon_pressure = max(
            0.0,
            min(1.0, self.carbon_pressure),
        )

        self.energy_pressure = max(
            0.0,
            min(1.0, self.energy_pressure),
        )

        self.resource_pressure = max(
            0.0,
            min(1.0, self.resource_pressure),
        )

        self.cost_pressure = max(
            0.0,
            min(1.0, self.cost_pressure),
        )

        self.schedule_pressure = max(
            0.0,
            min(1.0, self.schedule_pressure),
        )

        self.uncertainty = max(
            0.0,
            min(1.0, self.uncertainty),
        )

    # -----------------------------------------

    @classmethod
    def neutral(cls):

        """
        Neutral context.

        Used at initialization.
        """

        return cls(

            carbon_pressure=0.5,

            energy_pressure=0.5,

            resource_pressure=0.5,

            cost_pressure=0.5,

            schedule_pressure=0.5,

            uncertainty=0.5,

        )

    # -----------------------------------------

    def __str__(self):

        return (

            f"Context("

            f"carbon={self.carbon_pressure:.2f}, "

            f"energy={self.energy_pressure:.2f}, "

            f"resource={self.resource_pressure:.2f}, "

            f"cost={self.cost_pressure:.2f}, "

            f"schedule={self.schedule_pressure:.2f}, "

            f"uncertainty={self.uncertainty:.2f}"

            f")"

        )
