"""
project.py

Project model for the Context-Adaptive Sustainable
Multi-Mode Resource-Constrained Project Scheduling Problem.

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from copy import deepcopy

from dataclasses import dataclass
from dataclasses import field

from src.models.activity import Activity
from src.models.resource import Resource


@dataclass(slots=True)
class Project:
    """
    Complete scheduling instance.
    """

    # --------------------------------------------------
    # Basic Information
    # --------------------------------------------------

    name: str = ""

    instance_name: str = ""

    description: str = ""

    horizon: int = 0

    # --------------------------------------------------
    # Numbers from PSPLIB Header
    # --------------------------------------------------

    jobs: int = 0

    renewable_count: int = 0

    nonrenewable_count: int = 0

    doubly_count: int = 0

    # --------------------------------------------------
    # Capacities
    # --------------------------------------------------

    renewable_capacities: list[int] = field(
        default_factory=list
    )

    nonrenewable_capacities: list[int] = field(
        default_factory=list
    )

    doubly_capacities: list[int] = field(
        default_factory=list
    )

    # --------------------------------------------------
    # Resources
    # --------------------------------------------------

    renewable_resources: dict[int, Resource] = field(
        default_factory=dict
    )

    nonrenewable_resources: dict[int, Resource] = field(
        default_factory=dict
    )

    doubly_resources: dict[int, Resource] = field(
        default_factory=dict
    )

    # --------------------------------------------------
    # Activities
    # --------------------------------------------------

    activities: dict[int, Activity] = field(
        default_factory=dict
    )

    # --------------------------------------------------
    # Sustainability References
    # --------------------------------------------------

    baseline_cost: float = 1.0

    baseline_carbon: float = 1.0

    baseline_energy: float = 1.0

    reference_cost: float = 1.0

    reference_carbon: float = 1.0

    reference_energy: float = 1.0
    # =====================================================
    # Renewable Resources
    # =====================================================

    def add_renewable_resource(
        self,
        resource: Resource,
    ) -> None:

        self.renewable_resources[
            resource.id
        ] = resource

    def get_renewable_resource(
        self,
        resource_id: int,
    ) -> Resource:

        return self.renewable_resources[
            resource_id
        ]

    @property
    def renewable_resources_list(
        self,
    ) -> list[Resource]:

        return sorted(

            self.renewable_resources.values(),

            key=lambda r: r.id,

        )

    @property
    def renewable_resource_ids(
        self,
    ) -> list[int]:

        return sorted(

            self.renewable_resources.keys()

        )

    @property
    def number_of_renewable_resources(
        self,
    ) -> int:

        return len(

            self.renewable_resources

        )

    # =====================================================
    # Nonrenewable Resources
    # =====================================================

    def add_nonrenewable_resource(

        self,

        resource: Resource,

    ) -> None:

        self.nonrenewable_resources[
            resource.id
        ] = resource

    def get_nonrenewable_resource(

        self,

        resource_id: int,

    ) -> Resource:

        return self.nonrenewable_resources[
            resource_id
        ]

    @property
    def nonrenewable_resources_list(
        self,
    ) -> list[Resource]:

        return sorted(

            self.nonrenewable_resources.values(),

            key=lambda r: r.id,

        )

    @property
    def nonrenewable_resource_ids(
        self,
    ) -> list[int]:

        return sorted(

            self.nonrenewable_resources.keys()

        )

    @property
    def number_of_nonrenewable_resources(
        self,
    ) -> int:

        return len(

            self.nonrenewable_resources

        )

    # =====================================================
    # Doubly Constrained Resources
    # =====================================================

    def add_doubly_resource(

        self,

        resource: Resource,

    ) -> None:

        self.doubly_resources[
            resource.id
        ] = resource

    @property
    def doubly_resources_list(
        self,
    ) -> list[Resource]:

        return sorted(

            self.doubly_resources.values(),

            key=lambda r: r.id,

        )

    @property
    def number_of_doubly_constrained_resources(
        self,
    ) -> int:

        return len(

            self.doubly_resources

        )

    # =====================================================
    # All Resources
    # =====================================================

    @property
    def resources(
        self,
    ) -> list[Resource]:

        return (

            self.renewable_resources_list

            + self.nonrenewable_resources_list

            + self.doubly_resources_list

        )

    @property
    def total_resources(
        self,
    ) -> int:

        return len(

            self.resources

        )
    # =====================================================
    # Activities
    # =====================================================

    def add_activity(
        self,
        activity: Activity,
    ) -> None:

        self.activities[
            activity.id
        ] = activity

    def get_activity(
        self,
        activity_id: int,
    ) -> Activity:

        return self.activities[
            activity_id
        ]

    def has_activity(
        self,
        activity_id: int,
    ) -> bool:

        return activity_id in self.activities

    def remove_activity(
        self,
        activity_id: int,
    ) -> None:

        del self.activities[
            activity_id
        ]

    def clear_activities(
        self,
    ) -> None:

        self.activities.clear()

    @property
    def activities_list(
        self,
    ) -> list[Activity]:

        return sorted(

            self.activities.values(),

            key=lambda a: a.id,

        )

    @property
    def activity_ids(
        self,
    ) -> list[int]:

        return sorted(

            self.activities.keys()

        )

    @property
    def number_of_activities(
        self,
    ) -> int:

        return len(

            self.activities

        )

    @property
    def total_modes(
        self,
    ) -> int:

        return sum(

            activity.number_of_modes

            for activity in self.activities.values()

        )

    @property
    def start_activities(
        self,
    ) -> list[Activity]:

        return [

            activity

            for activity in self.activities.values()

            if activity.is_start_activity

        ]

    @property
    def finish_activities(
        self,
    ) -> list[Activity]:

        return [

            activity

            for activity in self.activities.values()

            if activity.is_finish_activity

        ]
        
            # =====================================================
    # Precedence
    # =====================================================

    def add_precedence(

        self,

        predecessor: int,

        successor: int,

    ) -> None:

        pred = self.get_activity(predecessor)

        succ = self.get_activity(successor)

        pred.add_successor(successor)

        succ.add_predecessor(predecessor)

    # -----------------------------------------------------

    def predecessors(

        self,

        activity_id: int,

    ) -> list[Activity]:

        activity = self.get_activity(activity_id)

        return [

            self.get_activity(i)

            for i in sorted(activity.predecessors)

        ]

    # -----------------------------------------------------

    def successors(

        self,

        activity_id: int,

    ) -> list[Activity]:

        activity = self.get_activity(activity_id)

        return [

            self.get_activity(i)

            for i in sorted(activity.successors)

        ]

    # -----------------------------------------------------

    @property

    def total_edges(

        self,

    ) -> int:

        return sum(

            len(a.successors)

            for a in self.activities.values()

        )

    # =====================================================
    # Cycle Detection
    # =====================================================

    def detect_cycle(

        self,

    ) -> bool:

        visited = set()

        stack = set()

        def dfs(node):

            visited.add(node)

            stack.add(node)

            activity = self.get_activity(node)

            for nxt in activity.successors:

                if nxt not in visited:

                    if dfs(nxt):

                        return True

                elif nxt in stack:

                    return True

            stack.remove(node)

            return False

        for activity in self.activities.values():

            if activity.id not in visited:

                if dfs(activity.id):

                    return True

        return False

    # =====================================================
    # Validation
    # =====================================================

    def validate(

        self,

    ) -> None:

        if self.number_of_activities == 0:

            raise ValueError("Project contains no activities.")

        if self.detect_cycle():

            raise ValueError("Project network contains a cycle.")

        for activity in self.activities.values():

            if not activity.has_modes:

                raise ValueError(

                    f"Activity {activity.id} has no modes."

                )

    # =====================================================
    # Topological Sort
    # =====================================================

    def topological_sort(

        self,

    ) -> list[int]:

        indegree = {

            a.id: a.indegree

            for a in self.activities.values()

        }

        queue = sorted(

            [

                i

                for i, d in indegree.items()

                if d == 0

            ]

        )

        order = []

        while queue:

            current = queue.pop(0)

            order.append(current)

            activity = self.get_activity(current)

            for successor in sorted(activity.successors):

                indegree[successor] -= 1

                if indegree[successor] == 0:

                    queue.append(successor)

                    queue.sort()

        if len(order) != self.number_of_activities:

            raise ValueError(

                "Cycle detected."

            )

        return order

    # -----------------------------------------------------

    @property

    def ordered_activities(

        self,

    ) -> list[Activity]:

        return [

            self.get_activity(i)

            for i in self.topological_sort()

        ]

    # =====================================================

    def reset_schedule(

        self,

    ) -> None:

        for activity in self.activities.values():

            activity.reset_schedule()
            
    # =====================================================
    # Sustainability Statistics
    # =====================================================

    def update_reference_values(
        self,
    ) -> None:
        """
        Compute reference sustainability values.

        These values are used for normalization.
        """

        total_cost = 0.0
        total_carbon = 0.0
        total_energy = 0.0

        for activity in self.activities.values():

            if not activity.modes:
                continue

            total_cost += max(
                mode.cost
                for mode in activity.modes
            )

            total_carbon += max(
                mode.carbon
                for mode in activity.modes
            )

            total_energy += max(
                mode.energy
                for mode in activity.modes
            )

        self.reference_cost = max(
            total_cost,
            1.0,
        )

        self.reference_carbon = max(
            total_carbon,
            1.0,
        )

        self.reference_energy = max(
            total_energy,
            1.0,
        )

        self.baseline_cost = self.reference_cost

        self.baseline_carbon = self.reference_carbon

        self.baseline_energy = self.reference_energy

    # =====================================================
    # Clone
    # =====================================================

    def clone(
        self,
    ) -> "Project":

        return deepcopy(self)

    # =====================================================
    # Export
    # =====================================================

    def to_dict(
        self,
    ) -> dict:

        return {

            "instance": self.instance_name,

            "activities": self.number_of_activities,

            "renewable": self.number_of_renewable_resources,

            "nonrenewable": self.number_of_nonrenewable_resources,

            "doubly": self.number_of_doubly_constrained_resources,

            "edges": self.total_edges,

            "modes": self.total_modes,

            "horizon": self.horizon,

            "reference_cost": self.reference_cost,

            "reference_carbon": self.reference_carbon,

            "reference_energy": self.reference_energy,

        }

    # =====================================================
    # Summary
    # =====================================================

    def summary(
        self,
    ) -> str:

        return (

            f"Project("
            f"activities={self.number_of_activities}, "
            f"modes={self.total_modes}, "
            f"renewable={self.number_of_renewable_resources}, "
            f"nonrenewable={self.number_of_nonrenewable_resources}, "
            f"doubly={self.number_of_doubly_constrained_resources}, "
            f"edges={self.total_edges}"
            f")"

        )

    # =====================================================

    def __str__(
        self,
    ) -> str:

        return self.summary()

    # =====================================================

    def __repr__(
        self,
    ) -> str:

        return self.summary()