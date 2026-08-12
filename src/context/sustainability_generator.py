"""
sustainability_generator.py

Generates sustainability indicators
(cost, carbon, energy)
for every execution mode.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from src.models.project import Project


class SustainabilityGenerator:
    """
    Generates synthetic sustainability indicators.

    These values are deterministic.

    They are used until real project data
    becomes available.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:

        self.project = project

    # ---------------------------------------------------------

    def generate(self) -> None:

        for activity in self.project.activities.values():

            if not activity.modes:
                print(f"Activity {activity.id} has NO MODES")
                continue
        
            positive_durations = [
                m.duration
                for m in activity.modes
                if m.duration > 0
            ]

            if positive_durations:

                max_duration = max(positive_durations)

                min_duration = min(positive_durations)

            else:
                # Dummy activity (duration = 0)

                max_duration = 1

                min_duration = 1

            for mode in activity.modes:
                
                duration = max(mode.duration, 1)

                speed_factor = max_duration / duration
                
                renewable = sum(mode.renewable)

                nonrenewable = sum(mode.nonrenewable)
                
                
                base_cost = 8000

                resource_cost = renewable * 700

                nonrenewable_cost = nonrenewable * 1300

                time_cost = duration * 250

                acceleration_cost = speed_factor * 900

                mode_factor = 1.0 + mode.id * 0.05

                activity_factor = 1.0 + activity.id * 0.01

                mode.cost = (

                    base_cost

                    + resource_cost

                    + nonrenewable_cost

                    + time_cost

                    + acceleration_cost

                )

                mode.cost *= mode_factor

                mode.cost *= activity_factor
                
                
                base_carbon = 200

                resource_carbon = renewable * 80

                nonrenewable_carbon = nonrenewable * 320

                duration_carbon = duration * 20

                speed_carbon = speed_factor * 60

                mode.carbon = (

                    base_carbon

                    + resource_carbon

                    + nonrenewable_carbon

                    + duration_carbon

                    + speed_carbon

                )

                mode.carbon *= (1.0 + mode.id * 0.03)

                mode.carbon *= (1.0 + activity.id * 0.005)
                
                
                base_energy = 300

                resource_energy = renewable * 120

                nonrenewable_energy = nonrenewable * 420

                duration_energy = duration * 18

                speed_energy = speed_factor * 55

                mode.energy = (

                    base_energy

                    + resource_energy

                    + nonrenewable_energy

                    + duration_energy

                    + speed_energy

                )

                mode.energy *= (1.0 + mode.id * 0.04)

                mode.energy *= (1.0 + activity.id * 0.006)