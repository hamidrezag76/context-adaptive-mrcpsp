"""
Global constants used throughout the CA-SMRCPSP framework.

Author:
CA-SMRCPSP Research Project

Description
-----------
This module centralizes all global constants, enumerations and
shared values used across the scheduling framework.

Keeping constants in a single location improves maintainability
and eliminates magic numbers.
"""

from __future__ import annotations

from enum import Enum


class Objective(str, Enum):
    """Optimization objectives."""

    MAKESPAN = "makespan"
    COST = "cost"
    CARBON = "carbon"
    ENERGY = "energy"


class ResourceType(str, Enum):
    """Resource categories."""

    RENEWABLE = "renewable"
    NON_RENEWABLE = "non_renewable"


class ScheduleStatus(str, Enum):
    """Schedule feasibility."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


class ContextDimension(str, Enum):
    """Project context categories."""

    RESOURCE = "resource"

    ENVIRONMENT = "environment"

    OPERATION = "operation"

    SUSTAINABILITY = "sustainability"

    MANAGEMENT = "management"


DEFAULT_RANDOM_SEED: int = 42

EPSILON: float = 1e-9

MAX_ITERATIONS: int = 1000

DEFAULT_POPULATION_SIZE: int = 100
