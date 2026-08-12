"""
statistical_analysis.py

Statistical comparison of paired multi-seed results
for Baseline NSGA-II and Context-Adaptive NSGA-II.

CA-SMRCPSP Research Project
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class StatisticalResult:
    """
    Statistical comparison for one performance metric.

    All comparisons are paired by seed.

    For HV:
        higher is better.

    For IGD+:
        lower is better.
    """

    metric: str

    baseline_mean: float
    adaptive_mean: float

    baseline_std: float
    adaptive_std: float

    mean_difference: float
    median_difference: float

    better_adaptive_count: int
    better_baseline_count: int
    ties: int

    paired_t_statistic: float
    paired_t_pvalue: float

    wilcoxon_statistic: float
    wilcoxon_pvalue: float

    cohens_d: float

    relative_improvement_percent: float

    @property
    def statistically_significant(self) -> bool:
        """
        Two-sided Wilcoxon significance at alpha=0.05.
        """
        return self.wilcoxon_pvalue < 0.05


class StatisticalAnalyzer:
    """
    Performs paired statistical analysis across seeds.
    """

    def __init__(
        self,
        alpha: float = 0.05,
    ) -> None:

        if not 0.0 < alpha < 1.0:
            raise ValueError(
                "Alpha must be between 0 and 1."
            )

        self.alpha = float(alpha)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def compare(
        self,
        baseline: Sequence[float],
        adaptive: Sequence[float],
        metric: str,
    ) -> StatisticalResult:

        baseline = np.asarray(
            baseline,
            dtype=float,
        )

        adaptive = np.asarray(
            adaptive,
            dtype=float,
        )

        if baseline.ndim != 1:
            raise ValueError(
                "Baseline values must be one-dimensional."
            )

        if adaptive.ndim != 1:
            raise ValueError(
                "Adaptive values must be one-dimensional."
            )

        if len(baseline) != len(adaptive):
            raise ValueError(
                "Baseline and adaptive results must "
                "contain the same number of paired observations."
            )

        if len(baseline) < 2:
            raise ValueError(
                "At least two paired observations are required."
            )

        if not np.all(
            np.isfinite(baseline)
        ):
            raise ValueError(
                "Baseline contains non-finite values."
            )

        if not np.all(
            np.isfinite(adaptive)
        ):
            raise ValueError(
                "Adaptive contains non-finite values."
            )

        metric = str(metric).strip().lower()

        if metric not in {
            "hypervolume",
            "hv",
            "igd_plus",
            "igd+",
        }:
            raise ValueError(
                "Unsupported metric. "
                "Use 'hypervolume' or 'igd_plus'."
            )

        differences = adaptive - baseline

        baseline_mean = float(
            np.mean(baseline)
        )

        adaptive_mean = float(
            np.mean(adaptive)
        )

        baseline_std = float(
            np.std(
                baseline,
                ddof=1,
            )
        )

        adaptive_std = float(
            np.std(
                adaptive,
                ddof=1,
            )
        )

        mean_difference = float(
            np.mean(differences)
        )

        median_difference = float(
            np.median(differences)
        )

        better_adaptive_count, better_baseline_count, ties = (
            self._count_direction(
                baseline,
                adaptive,
                metric,
            )
        )

        t_statistic, t_pvalue = self._paired_ttest(
            baseline,
            adaptive,
        )

        wilcoxon_statistic, wilcoxon_pvalue = (
            self._wilcoxon(
                baseline,
                adaptive,
            )
        )

        cohens_d = self._cohens_d(
            differences
        )

        relative_improvement = (
            self._relative_improvement(
                baseline_mean,
                adaptive_mean,
                metric,
            )
        )

        return StatisticalResult(
            metric=metric,

            baseline_mean=baseline_mean,
            adaptive_mean=adaptive_mean,

            baseline_std=baseline_std,
            adaptive_std=adaptive_std,

            mean_difference=mean_difference,
            median_difference=median_difference,

            better_adaptive_count=(
                better_adaptive_count
            ),

            better_baseline_count=(
                better_baseline_count
            ),

            ties=ties,

            paired_t_statistic=t_statistic,
            paired_t_pvalue=t_pvalue,

            wilcoxon_statistic=(
                wilcoxon_statistic
            ),

            wilcoxon_pvalue=(
                wilcoxon_pvalue
            ),

            cohens_d=cohens_d,

            relative_improvement_percent=(
                relative_improvement
            ),
        )

    # ---------------------------------------------------------
    # Paired t-test
    # ---------------------------------------------------------

    @staticmethod
    def _paired_ttest(
        baseline: np.ndarray,
        adaptive: np.ndarray,
    ) -> tuple[float, float]:

        result = ttest_rel(
            adaptive,
            baseline,
        )

        statistic = float(
            result.statistic
        )

        pvalue = float(
            result.pvalue
        )

        if not np.isfinite(statistic):
            statistic = 0.0

        if not np.isfinite(pvalue):
            pvalue = 1.0

        return statistic, pvalue

    # ---------------------------------------------------------
    # Wilcoxon signed-rank
    # ---------------------------------------------------------

    def _wilcoxon(
        self,
        baseline: np.ndarray,
        adaptive: np.ndarray,
    ) -> tuple[float, float]:

        differences = adaptive - baseline

        nonzero = differences[
            differences != 0.0
        ]

        if len(nonzero) == 0:
            return 0.0, 1.0

        try:

            result = wilcoxon(
                adaptive,
                baseline,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

        except ValueError:

            return 0.0, 1.0

        statistic = float(
            result.statistic
        )

        pvalue = float(
            result.pvalue
        )

        if not np.isfinite(statistic):
            statistic = 0.0

        if not np.isfinite(pvalue):
            pvalue = 1.0

        return statistic, pvalue

    # ---------------------------------------------------------
    # Direction
    # ---------------------------------------------------------

    @staticmethod
    def _count_direction(
        baseline: np.ndarray,
        adaptive: np.ndarray,
        metric: str,
    ) -> tuple[int, int, int]:

        better_adaptive = 0
        better_baseline = 0
        ties = 0

        for b, a in zip(
            baseline,
            adaptive,
        ):

            if metric in {
                "hypervolume",
                "hv",
            }:

                if a > b:
                    better_adaptive += 1

                elif b > a:
                    better_baseline += 1

                else:
                    ties += 1

            else:

                if a < b:
                    better_adaptive += 1

                elif b < a:
                    better_baseline += 1

                else:
                    ties += 1

        return (
            better_adaptive,
            better_baseline,
            ties,
        )

    # ---------------------------------------------------------
    # Cohen's d for paired samples
    # ---------------------------------------------------------

    @staticmethod
    def _cohens_d(
        differences: np.ndarray,
    ) -> float:
        """
        Cohen's d for paired observations.

        The effect size is computed from the
        paired differences.

        When the standard deviation of the
        differences is effectively zero, the
        standardized effect is mathematically
        unbounded.
        """

        if len(differences) < 2:
            return 0.0

        mean_difference = float(
            np.mean(differences)
        )

        standard_deviation = float(
            np.std(
                differences,
                ddof=1,
            )
        )

        tolerance = 1e-12

        if standard_deviation <= tolerance:

            if abs(mean_difference) <= tolerance:
                return 0.0

            return float(
                np.copysign(
                    np.inf,
                    mean_difference,
                )
            )

        return float(
            mean_difference
            / standard_deviation
        )

    # ---------------------------------------------------------
    # Relative improvement
    # ---------------------------------------------------------

    @staticmethod
    def _relative_improvement(
        baseline_mean: float,
        adaptive_mean: float,
        metric: str,
    ) -> float:

        if baseline_mean == 0.0:
            return 0.0

        if metric in {
            "hypervolume",
            "hv",
        }:

            return (
                (
                    adaptive_mean
                    - baseline_mean
                )
                / abs(baseline_mean)
            ) * 100.0

        return (
            (
                baseline_mean
                - adaptive_mean
            )
            / abs(baseline_mean)
        ) * 100.0
