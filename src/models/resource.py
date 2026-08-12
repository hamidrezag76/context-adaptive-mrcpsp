"""
resource.py

Domain model representing project resources for the
Context-Adaptive Sustainable Multi-Mode RCPSP framework.

Author
------
CA-SMRCPSP Research Project

Description
-----------
Each resource belongs to one of two categories:

    • Renewable
    • Non-renewable

The class is intentionally immutable to avoid accidental
modification during optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.constants import ResourceType


@dataclass(frozen=True, slots=True)
class Resource:
    """
    Represents one project resource.

    Parameters
    ----------
    id:
        Unique resource identifier.

    name:
        Human-readable name.

    resource_type:
        Renewable or Non-renewable.

    capacity:
        Available quantity.
    """

    id: int

    name: str

    resource_type: ResourceType

    capacity: float

    @property
    def is_renewable(self) -> bool:
        """Return True if resource is renewable."""
        return self.resource_type == ResourceType.RENEWABLE

    @property
    def is_nonrenewable(self) -> bool:
        """Return True if resource is non-renewable."""
        return self.resource_type == ResourceType.NON_RENEWABLE

    def __str__(self) -> str:
        return (
            f"Resource("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"type={self.resource_type.value}, "
            f"capacity={self.capacity}"
            f")"
        )
