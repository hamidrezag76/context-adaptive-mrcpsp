"""
repair.py

Priority List Repair Operator

Repairs infeasible priority lists while
preserving precedence constraints.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from collections import deque

from src.models.project import Project


class RepairOperator:
    """
    Repairs priority lists.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:

        self.project = project

    def repair(
        self,
        priority_list: list[int],
    ) -> list[int]:
        """
        Repair priority list using precedence constraints.
        """

        position = {act: i for i, act in enumerate(priority_list)}

        indegree = {}

        graph = {}

        for activity in self.project.activities.values():

            indegree[activity.id] = 0
            graph[activity.id] = []

        for activity in self.project.activities.values():

            for successor in activity.successors:

                graph[activity.id].append(successor)

                indegree[successor] += 1

        available = []

        for act in indegree:

            if indegree[act] == 0:

                available.append(act)

        repaired = []

        while available:

            available.sort(
                key=lambda x: position[x],
            )

            current = available.pop(0)

            repaired.append(current)

            for successor in graph[current]:

                indegree[successor] -= 1

                if indegree[successor] == 0:

                    available.append(successor)

        return repaired
