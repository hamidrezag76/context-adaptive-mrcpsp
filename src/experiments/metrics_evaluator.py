"""
metrics_evaluator.py

Multi-seed comparative evaluator for
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


class MultiSeedMetricsEvaluator:
    """
    Computes statistically comparable HV and IGD+ metrics
    across multiple stochastic optimization runs.

    A single common normalization and reference construction
    is used for all baseline and adaptive archives across
    all supplied seeds.
    """

    def __init__(
        self,
        baseline_archives: dict[int, list[ObjectiveVector]],
        adaptive_archives: dict[int, list[ObjectiveVector]],
    ) -> None:

        if not baseline_archives:
            raise ValueError(
                "Baseline archives cannot be empty."
            )

        if not adaptive_archives:
            raise ValueError(
                "Adaptive archives cannot be empty."
            )

        if set(baseline_archives) != set(
            adaptive_archives
        ):
            raise ValueError(
                "Baseline and adaptive seed sets "
                "must be identical."
            )

        self.seeds = sorted(
            baseline_archives
        )

        self.baseline_archives = {
            seed: [
                tuple(float(x) for x in point)
                for point in points
            ]
            for seed, points
            in baseline_archives.items()
        }

        self.adaptive_archives = {
            seed: [
                tuple(float(x) for x in point)
                for point in points
            ]
            for seed, points
            in adaptive_archives.items()
        }

        # -----------------------------------------------------
        # Validate archives
        # -----------------------------------------------------

        for seed in self.seeds:

            if not self.baseline_archives[seed]:
                raise ValueError(
                    f"Baseline archive for seed {seed} "
                    "is empty."
                )

            if not self.adaptive_archives[seed]:
                raise ValueError(
                    f"Adaptive archive for seed {seed} "
                    "is empty."
                )

        # -----------------------------------------------------
        # Complete experimental union
        # -----------------------------------------------------

        all_points = []

        for seed in self.seeds:

            all_points.extend(
                self.baseline_archives[seed]
            )

            all_points.extend(
                self.adaptive_archives[seed]
            )

        if not all_points:
            raise ValueError(
                "Combined experimental archive is empty."
            )

        self.all_points = all_points

        # -----------------------------------------------------
        # Common normalization
        # -----------------------------------------------------

        self.normalizer = (
            ObjectiveNormalizer.from_sets(
                *[
                    self.baseline_archives[seed]
                    for seed in self.seeds
                ],
                *[
                    self.adaptive_archives[seed]
                    for seed in self.seeds
                ],
            )
        )

        self.baseline_normalized = {}

        self.adaptive_normalized = {}

        for seed in self.seeds:

            self.baseline_normalized[seed] = (
                self.normalizer.normalize_set(
                    self.baseline_archives[seed]
                )
            )

            self.adaptive_normalized[seed] = (
                self.normalizer.normalize_set(
                    self.adaptive_archives[seed]
                )
            )

        # -----------------------------------------------------
        # Complete normalized union
        # -----------------------------------------------------

        normalized_union = []

        for seed in self.seeds:

            normalized_union.extend(
                self.baseline_normalized[seed]
            )

            normalized_union.extend(
                self.adaptive_normalized[seed]
            )

        self.normalized_union = normalized_union

        # -----------------------------------------------------
        # Common nondominated reference set
        #
        # Used by IGD+.
        # -----------------------------------------------------

        self.reference_set = (
            ReferenceSetBuilder.nondominated(
                self.normalized_union
            )
        )

        if not self.reference_set:
            raise ValueError(
                "Common nondominated reference set "
                "is empty."
            )

        # -----------------------------------------------------
        # Common HV reference point
        #
        # Constructed from ALL normalized points.
        # -----------------------------------------------------

        self.reference_point = (
            ReferenceSetBuilder.reference_point(
                self.normalized_union,
                margin=0.05,
            )
        )

    # ---------------------------------------------------------
    # Per-seed evaluation
    # ---------------------------------------------------------

    def evaluate_seed(
        self,
        seed: int,
    ) -> dict[str, float]:

        if seed not in self.seeds:
            raise KeyError(
                f"Unknown seed: {seed}"
            )

        hypervolume = Hypervolume(
            self.reference_point
        )

        igd_plus = IGDPlus(
            self.reference_set
        )

        baseline_hv = hypervolume.compute(
            self.baseline_normalized[seed]
        )

        adaptive_hv = hypervolume.compute(
            self.adaptive_normalized[seed]
        )

        baseline_igd = igd_plus.compute(
            self.baseline_normalized[seed]
        )

        adaptive_igd = igd_plus.compute(
            self.adaptive_normalized[seed]
        )

        return {
            "seed": float(seed),
            "baseline_hv": baseline_hv,
            "adaptive_hv": adaptive_hv,
            "baseline_igd_plus": baseline_igd,
            "adaptive_igd_plus": adaptive_igd,
        }

    # ---------------------------------------------------------
    # All seeds
    # ---------------------------------------------------------

    def evaluate(
        self,
    ) -> list[dict[str, float]]:

        return [
            self.evaluate_seed(seed)
            for seed in self.seeds
        ]

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    @staticmethod
    def _mean(
        values: list[float],
    ) -> float:

        if not values:
            raise ValueError(
                "Cannot compute mean of empty values."
            )

        return (
            sum(values)
            / len(values)
        )

    @staticmethod
    def _sample_std(
        values: list[float],
    ) -> float:

        if len(values) < 2:
            return 0.0

        mean = (
            sum(values)
            / len(values)
        )

        variance = (
            sum(
                (value - mean) ** 2
                for value in values
            )
            / (len(values) - 1)
        )

        return variance ** 0.5

    @staticmethod
    def _relative_improvement(
        baseline: float,
        adaptive: float,
        higher_is_better: bool,
    ) -> float:

        if baseline == 0.0:
            return 0.0

        if higher_is_better:

            return (
                (adaptive - baseline)
                / abs(baseline)
            ) * 100.0

        return (
            (baseline - adaptive)
            / abs(baseline)
        ) * 100.0

    def summary(
        self,
    ) -> dict[str, dict[str, float]]:

        results = self.evaluate()

        baseline_hv = [
            r["baseline_hv"]
            for r in results
        ]

        adaptive_hv = [
            r["adaptive_hv"]
            for r in results
        ]

        baseline_igd = [
            r["baseline_igd_plus"]
            for r in results
        ]

        adaptive_igd = [
            r["adaptive_igd_plus"]
            for r in results
        ]

        baseline_hv_mean = self._mean(
            baseline_hv
        )

        adaptive_hv_mean = self._mean(
            adaptive_hv
        )

        baseline_igd_mean = self._mean(
            baseline_igd
        )

        adaptive_igd_mean = self._mean(
            adaptive_igd
        )

        return {

            "hypervolume": {

                "baseline_mean":
                    baseline_hv_mean,

                "baseline_std":
                    self._sample_std(
                        baseline_hv
                    ),

                "adaptive_mean":
                    adaptive_hv_mean,

                "adaptive_std":
                    self._sample_std(
                        adaptive_hv
                    ),

                "relative_improvement_percent":
                    self._relative_improvement(
                        baseline_hv_mean,
                        adaptive_hv_mean,
                        higher_is_better=True,
                    ),
            },

            "igd_plus": {

                "baseline_mean":
                    baseline_igd_mean,

                "baseline_std":
                    self._sample_std(
                        baseline_igd
                    ),

                "adaptive_mean":
                    adaptive_igd_mean,

                "adaptive_std":
                    self._sample_std(
                        adaptive_igd
                    ),

                "relative_improvement_percent":
                    self._relative_improvement(
                        baseline_igd_mean,
                        adaptive_igd_mean,
                        higher_is_better=False,
                    ),
            },
        }
