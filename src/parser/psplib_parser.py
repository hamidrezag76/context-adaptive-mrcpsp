"""
psplib_parser.py

Production-ready PSPLIB parser for the
Context-Adaptive Sustainable Multi-Mode
Resource-Constrained Project Scheduling Problem
(CA-SMRCPSP).

This parser uses the official psplib package
instead of manually parsing benchmark files.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from pathlib import Path

from psplib import parse_psplib

from src.models.activity import Activity
from src.models.mode import Mode
from src.models.project import Project
from src.models.resource import Resource
from src.parser.parser import BaseParser
from src.utils.constants import ResourceType


class PSPLIBParser(BaseParser):
    """
    Parser for standard PSPLIB benchmark instances.
    """

    def supports(
        self,
        file_path: str | Path,
    ) -> bool:
        """
        Returns True if the parser supports
        the specified file.
        """

        return Path(file_path).suffix.lower() == ".mm"

    def load(
        self,
        file_path: str | Path,
    ) -> Project:
        """
        Read one PSPLIB instance and convert it
        into an internal Project object.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(path)

        if not self.supports(path):
            raise ValueError(f"Unsupported file type: {path.suffix}")

        instance = parse_psplib(path)

        project = Project(
            name=path.stem,
            instance_name=path.name,
        )

        self._convert_resources(
            instance,
            project,
        )

        self._convert_activities(
            instance,
            project,
        )

        self._convert_precedence(
            instance,
            project,
        )

        project.validate()

        return project

    # --------------------------------------------------
    # Resources
    # --------------------------------------------------

    def _convert_resources(
        self,
        instance,
        project: Project,
    ) -> None:
        """
        Convert PSPLIB resources into the project model.

        Resource capacities are stored both in the
        resource dictionaries and in the corresponding
        capacity lists used by the scheduling engine.
        """

        project.renewable_capacities.clear()
        project.nonrenewable_capacities.clear()
        project.doubly_capacities.clear()

        project.renewable_count = 0
        project.nonrenewable_count = 0
        project.doubly_count = 0

        for index, resource in enumerate(
            instance.resources,
            start=1,
        ):

            new_resource = Resource(
                id=index,
                name=f"R{index}",
                resource_type=(
                    ResourceType.RENEWABLE
                    if resource.renewable
                    else ResourceType.NON_RENEWABLE
                ),
                capacity=resource.capacity,
            )

            if resource.renewable:

                project.add_renewable_resource(
                    new_resource
                )

                project.renewable_capacities.append(
                    int(resource.capacity)
                )

                project.renewable_count += 1

            else:

                project.add_nonrenewable_resource(
                    new_resource
                )

                project.nonrenewable_capacities.append(
                    int(resource.capacity)
                )

                project.nonrenewable_count += 1

    # --------------------------------------------------
    # Activities
    # --------------------------------------------------

    def _convert_activities(
        self,
        instance,
        project: Project,
    ) -> None:
        """
        Convert all activities and modes.
        """

        renewable_count = len(project.renewable_resources)

        for activity_id, activity_data in enumerate(
            instance.activities
        ):

            activity = Activity(
                id=activity_id
            )

            for mode_id, mode_data in enumerate(
                activity_data.modes,
                start=1,
            ):

                renewable_resources = []
                nonrenewable_resources = []

                for resource_index, demand in enumerate(
                    mode_data.demands,
                    start=1,
                ):

                    if resource_index <= renewable_count:

                        renewable_resources.append(
                            int(demand)
                        )

                    else:

                        nonrenewable_resources.append(
                            int(demand)
                        )

                total_resources = sum(
                    mode_data.demands
                )

                cost = (
                    mode_data.duration * 100
                    + total_resources * 25
                )

                carbon = (
                    mode_data.duration * 8
                    + total_resources * 2
                )

                energy = (
                    mode_data.duration * 15
                    + total_resources * 5
                )

                mode = Mode(
                    id=mode_id,
                    duration=mode_data.duration,
                    cost=cost,
                    carbon=carbon,
                    energy=energy,
                    renewable=renewable_resources,
                    nonrenewable=nonrenewable_resources,
                )

                activity.add_mode(
                    mode
                )

            if activity.modes:

                activity.selected_mode = (
                    activity.modes[0].id
                )

            project.add_activity(
                activity
            )

    # --------------------------------------------------
    # Precedence Relations
    # --------------------------------------------------

    def _convert_precedence(
        self,
        instance,
        project: Project,
    ) -> None:
        """
        Convert precedence relations.
        """

        for predecessor_id, activity in enumerate(instance.activities):

            for successor_id in activity.successors:

                project.add_precedence(
                    predecessor=predecessor_id,
                    successor=successor_id,
                )

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return f"{self.__class__.__name__}()"
