from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.models.project import Project


@dataclass(slots=True)
class ValidationResult:
    """
    Result of project validation.
    """

    valid: bool
    errors: List[str]
    warnings: List[str]


class ProjectValidator:
    """
    Validate parsed PSPLIB projects before scheduling.

    Checks include

    • duplicate activity ids
    • activities without modes
    • invalid predecessors
    • cyclic precedence
    • isolated activities
    """

    def validate(self, project: Project) -> ValidationResult:

        errors: list[str] = []
        warnings: list[str] = []

        self._check_duplicate_ids(project, errors)

        self._check_modes(project, errors)

        self._check_successors(project, errors)

        self._check_isolated(project, warnings)

        self._check_cycles(project, errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # -------------------------------------------------------------

    def _check_duplicate_ids(self, project, errors):

        ids = [a.id for a in project.activities.values()]

        if len(ids) != len(set(ids)):
            errors.append("Duplicate activity ids detected.")

    # -------------------------------------------------------------

    def _check_modes(self, project, errors):

        for activity in project.activities.values():

            if len(activity.modes) == 0:
                errors.append(
                    f"Activity {activity.id} has no execution mode."
                )

    # -------------------------------------------------------------

    def _check_successors(self, project, errors):

        activity_ids = {a.id for a in project.activities.values()}

        for activity in project.activities.values():

            for suc in activity.successors:

                if suc not in activity_ids:

                    errors.append(
                        f"Activity {activity.id} references unknown successor {suc}"
                    )

    # -------------------------------------------------------------

    def _check_isolated(self, project, warnings):

        referenced = set()

        for activity in project.activities.values():

            referenced.update(activity.successors)

        for activity in project.activities.values():

            if (
                len(activity.successors) == 0
                and activity.id not in referenced
            ):

                warnings.append(
                    f"Activity {activity.id} is isolated."
                )

    # -------------------------------------------------------------

    def _check_cycles(self, project, errors):

        WHITE = 0
        GRAY = 1
        BLACK = 2

        activity_ids = {a.id for a in project.activities.values()}

        color = {i: WHITE for i in activity_ids}

        adjacency = {}

        for activity in project.activities.values():
            adjacency[activity.id] = [
                s for s in activity.successors
                if s in activity_ids
            ]

        def dfs(node):

            color[node] = GRAY

            for nxt in adjacency.get(node, []):

                if color[nxt] == GRAY:
                    return True

                if color[nxt] == WHITE:
                    if dfs(nxt):
                        return True

            color[node] = BLACK
            return False

        for node in activity_ids:

            if color[node] == WHITE:

                if dfs(node):

                    errors.append("Cycle detected in precedence graph.")
                    return
