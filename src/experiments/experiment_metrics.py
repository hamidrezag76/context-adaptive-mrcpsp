"""
experiment_metrics.py

Comparative multi-objective metrics for
Baseline NSGA-II and Context-Adaptive NSGA-II.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from typing import Sequence

from src.metrics.hypervolume import Hypervolume
from src.metrics.igd_plus import IGDPlus

from src.experiments.objective_normalizer import (
    ObjectiveNormalizer,
)

from src.experiments.reference_set import (
    ReferenceSetBuilder,
)


ObjectiveVector = Sequence[float]


class ExperimentMetrics:
    """
    Computes comparable HV and IGD+ values for
    multiple approximation sets.

    Methodological separation:

        1. Common nondominated reference set
           -> used by IGD+.

        2. Common HV reference point
           -> constructed from the complete union of
              normalized approximation sets.

    This prevents the HV reference point from becoming
    artificially optimistic when extreme points are removed
    by nondominated filtering.
    """

    def __init__(
        self,
        baseline: list[ObjectiveVector],
        adaptive: list[ObjectiveVector],
    ) -> None:

        if not baseline:
            raise ValueError(
                "Baseline approximation set cannot be empty."
            )

        if not adaptive:
            raise ValueError(
                "Adaptive approximation set cannot be empty."
            )

        self.baseline = [
            tuple(float(x) for x in point)
            for point in baseline
        ]

        self.adaptive = [
            tuple(float(x) for x in point)
            for point in adaptive
        ]

        # -----------------------------------------------------
        # Common normalization
        # -----------------------------------------------------

        self.normalizer = (
            ObjectiveNormalizer.from_sets(
                self.baseline,
                self.adaptive,
            )
        )

        self.baseline_normalized = (
            self.normalizer.normalize_set(
                self.baseline
            )
        )

        self.adaptive_normalized = (
            self.normalizer.normalize_set(
                self.adaptive
            )
        )

        # -----------------------------------------------------
        # Common combined normalized set
        # -----------------------------------------------------

        self.combined_normalized = (
            self.baseline_normalized
            + self.adaptive_normalized
        )

        if not self.combined_normalized:
            raise ValueError(
                "Combined normalized approximation set is empty."
            )

        # -----------------------------------------------------
        # Common nondominated reference set
        #
        # Used ONLY for IGD+.
        # -----------------------------------------------------

        self.reference_set = (
            ReferenceSetBuilder.nondominated(
                self.combined_normalized
            )
        )

        if not self.reference_set:
            raise ValueError(
                "Common reference set is empty."
            )

        # -----------------------------------------------------
        # Common HV reference point
        #
        # IMPORTANT:
        #
        # The HV reference point must be worse than every
        # normalized point in BOTH approximation sets.
        #
        # Therefore it is constructed from the complete
        # combined set, NOT from the nondominated reference
        # set.
        # -----------------------------------------------------

        self.reference_point = (
            ReferenceSetBuilder.reference_point(
                self.combined_normalized,
                margin=0.05,
            )
        )

    # ---------------------------------------------------------
    # Hypervolume
    # ---------------------------------------------------------

    def hypervolume_baseline(
        self,
    ) -> float:

        metric = Hypervolume(
            self.reference_point
        )

        return metric.compute(
            self.baseline_normalized
        )

    # ---------------------------------------------------------

    def hypervolume_adaptive(
        self,
    ) -> float:

        metric = Hypervolume(
            self.reference_point
        )

        return metric.compute(
            self.adaptive_normalized
        )

    # ---------------------------------------------------------
    # IGD+
    # ---------------------------------------------------------

    def igd_plus_baseline(
        self,
    ) -> float:

        metric = IGDPlus(
            self.reference_set
        )

        return metric.compute(
            self.baseline_normalized
        )

    # ---------------------------------------------------------

    def igd_plus_adaptive(
        self,
    ) -> float:

        metric = IGDPlus(
            self.reference_set
        )

        return metric.compute(
            self.adaptive_normalized
        )

    # ---------------------------------------------------------
    # All metrics
    # ---------------------------------------------------------

    def evaluate(
        self,
    ) -> dict[str, float]:

        return {
            "baseline_hv":
                self.hypervolume_baseline(),

            "adaptive_hv":
                self.hypervolume_adaptive(),

            "baseline_igd_plus":
                self.igd_plus_baseline(),

            "adaptive_igd_plus":
                self.igd_plus_adaptive(),
        }
