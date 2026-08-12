"""
activity.py

Activity domain model for the Context-Adaptive Sustainable
Multi-Mode Resource-Constrained Project Scheduling Problem.

This class represents one project activity together with its
available execution modes and precedence relationships.

The class is intentionally independent from optimization
algorithms. It serves as the common data model shared by

    • Parser
    • Project
    • SSGS
    • Decoder
    • NSGA-II
    • Evaluation

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models.mode import Mode


@dataclass(slots=True)
class Activity:
    """
    Represents one project activity.

    Parameters
    ----------
    id
        Unique activity identifier.

    name
        Optional activity name.

    modes
        Available execution modes.

    predecessors
        Immediate predecessor activities.

    successors
        Immediate successor activities.
    """

    id: int

    name: str = ""

    modes: list[Mode] = field(default_factory=list)

    predecessors: set[int] = field(default_factory=set)

    successors: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:

        if self.id < 0:
            raise ValueError("Activity ID cannot be negative.")

    # ---------------------------------------------------------
    # Mode Management
    # ---------------------------------------------------------

    def add_mode(
        self,
        mode: Mode,
    ) -> None:
        """
        Add one execution mode.

        Raises
        ------
        ValueError
            If mode already exists.
        """

        for existing in self.modes:

            if existing.id == mode.id:

                raise ValueError(f"Mode {mode.id} already exists.")

        self.modes.append(mode)

    def get_mode(
        self,
        mode_id: int,
    ) -> Mode:
        """
        Return execution mode by ID.
        """

        for mode in self.modes:

            if mode.id == mode_id:
                return mode

        raise KeyError(f"Mode {mode_id} not found.")

    @property
    def number_of_modes(
        self,
    ) -> int:
        """
        Number of feasible modes.
        """

        return len(self.modes)
        # ---------------------------------------------------------

    # Precedence Relationships
    # ---------------------------------------------------------

    def add_predecessor(
        self,
        activity_id: int,
    ) -> None:
        """
        Add one predecessor activity.

        Parameters
        ----------
        activity_id
            Activity identifier.
        """

        if activity_id == self.id:
            raise ValueError("An activity cannot precede itself.")

        self.predecessors.add(activity_id)

    def add_successor(
        self,
        activity_id: int,
    ) -> None:
        """
        Add one successor activity.
        """

        if activity_id == self.id:
            raise ValueError("An activity cannot succeed itself.")

        self.successors.add(activity_id)

    def remove_predecessor(
        self,
        activity_id: int,
    ) -> None:

        self.predecessors.discard(activity_id)

    def remove_successor(
        self,
        activity_id: int,
    ) -> None:

        self.successors.discard(activity_id)

    @property
    def indegree(self) -> int:
        """
        Number of immediate predecessors.
        """

        return len(self.predecessors)

    @property
    def outdegree(self) -> int:
        """
        Number of immediate successors.
        """

        return len(self.successors)

    @property
    def is_start_activity(self) -> bool:
        """
        True if activity has no predecessors.
        """

        return self.indegree == 0

    @property
    def is_finish_activity(self) -> bool:
        """
        True if activity has no successors.
        """

        return self.outdegree == 0
        # ---------------------------------------------------------

    # Scheduling State
    # ---------------------------------------------------------

    selected_mode: int | None = None

    scheduled: bool = False

    start_time: int = 0

    finish_time: int = 0

    earliest_start: int = 0

    earliest_finish: int = 0

    latest_start: int = 0

    latest_finish: int = 0

    total_float: int = 0

    free_float: int = 0

    def reset_schedule(self) -> None:
        """
        Reset scheduling information.
        """

        self.selected_mode = None

        self.scheduled = False

        self.start_time = 0

        self.finish_time = 0

        self.earliest_start = 0

        self.earliest_finish = 0

        self.latest_start = 0

        self.latest_finish = 0

        self.total_float = 0

        self.free_float = 0
        # ---------------------------------------------------------

    # Helper Properties
    # ---------------------------------------------------------

    @property
    def has_modes(self) -> bool:

        return len(self.modes) > 0

    @property
    def is_scheduled(self) -> bool:

        return self.scheduled

    @property
    def duration(self) -> int:
        """
        Duration of currently selected mode.
        """

        if self.selected_mode is None:
            raise RuntimeError("No execution mode has been selected.")

        return self.get_mode(self.selected_mode).duration
