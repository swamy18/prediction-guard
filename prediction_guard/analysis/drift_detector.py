"""
Statistical drift detection algorithms for Prediction Guard.

This module provides stateless, pure functions for detecting various
types of drift in ML model predictions. Each function is:
- Stateless: No side effects, deterministic output
- Validated: Handles edge cases gracefully
- Documented: Clear mathematical definitions
- Tested: Comprehensive unit tests

Mathematical Background:
- KS Test: Kolmogorov-Smirnov test for distribution comparison
- PSI: Population Stability Index from credit risk modeling
- Cosine Distance: Geometric distance in embedding space
- Shannon Entropy: Information-theoretic uncertainty measure

Author: Prediction Guard Team
"""

from __future__ import annotations

import logging
import warnings

from typing import TYPE_CHECKING, Final, cast

import numpy as np

from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum samples for statistical validity
MIN_SAMPLES_KS_TEST: Final[int] = 5
MIN_SAMPLES_PSI: Final[int] = 10

# Small epsilon for numerical stability
EPSILON: Final[float] = 1e-10

# PSI interpretation thresholds (industry standard)
PSI_NO_CHANGE: Final[float] = 0.10
PSI_MODERATE_CHANGE: Final[float] = 0.25


# =============================================================================
# DRIFT DETECTOR CLASS
# =============================================================================

class DriftDetector:
    """
    Stateless drift detection algorithms.

    All methods are static and pure - they take data in and return
    metrics out, with no side effects.

    Usage:
        >>> from prediction_guard.analysis import DriftDetector
        >>> ks_stat, p_value = DriftDetector.ks_test(current, baseline)
        >>> psi = DriftDetector.population_stability_index(current_dist, base_dist)
    """

    # -------------------------------------------------------------------------
    # Distribution Comparison: KS Test
    # -------------------------------------------------------------------------

    @staticmethod
    def ks_test(
        current_values: ArrayLike,
        baseline_values: ArrayLike,
    ) -> tuple[float, float]:
        """
        Two-sample Kolmogorov-Smirnov test for distribution comparison.

        The KS test measures the maximum distance between two empirical
        cumulative distribution functions. It is sensitive to both
        location and shape differences.

        Mathematical Definition:
            D = max|F_n(x) - F_m(x)|
            where F_n and F_m are the empirical CDFs

        Args:
            current_values: Values from the current analysis window
            baseline_values: Values from the baseline period

        Returns:
            Tuple of (ks_statistic, p_value):
            - ks_statistic: Maximum distance between CDFs [0, 1]
            - p_value: Probability of observing this distance under H0

        Note:
            Returns (0.0, 1.0) if either sample is too small for
            statistical validity.

        Example:
            >>> baseline = [0.5, 0.6, 0.7, 0.8, 0.9]
            >>> current = [0.7, 0.8, 0.9, 1.0, 1.1]
            >>> stat, pval = DriftDetector.ks_test(current, baseline)
        """
        # Convert to numpy arrays and validate
        current = np.asarray(current_values, dtype=np.float64).ravel()
        baseline = np.asarray(baseline_values, dtype=np.float64).ravel()

        # Check minimum sample sizes
        if len(current) < MIN_SAMPLES_KS_TEST or len(baseline) < MIN_SAMPLES_KS_TEST:
            logger.debug(
                f"Insufficient samples for KS test: "
                f"current={len(current)}, baseline={len(baseline)}"
            )
            return 0.0, 1.0

        # Handle identical distributions (common in testing)
        if np.array_equal(current, baseline):
            return 0.0, 1.0

        # Remove NaN values with warning
        current_clean = current[~np.isnan(current)]
        baseline_clean = baseline[~np.isnan(baseline)]

        if len(current_clean) < len(current) or len(baseline_clean) < len(baseline):
            logger.warning(
                f"NaN values removed: current {len(current) - len(current_clean)}, "
                f"baseline {len(baseline) - len(baseline_clean)}"
            )

        if len(current_clean) < MIN_SAMPLES_KS_TEST or len(baseline_clean) < MIN_SAMPLES_KS_TEST:
            return 0.0, 1.0

        # Perform KS test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            statistic, p_value = stats.ks_2samp(current_clean, baseline_clean)

        return float(statistic), float(p_value)

    # -------------------------------------------------------------------------
    # Categorical Distribution: Population Stability Index
    # -------------------------------------------------------------------------

    @staticmethod
    def population_stability_index(
        current_distribution: dict[str, float],
        baseline_distribution: dict[str, float],
        epsilon: float = EPSILON,
    ) -> float:
        """
        Population Stability Index (PSI) for categorical distribution shift.

        PSI is a symmetric measure of how much a distribution has shifted.
        Originally from credit risk modeling, it's widely used for
        monitoring categorical prediction distributions.

        Mathematical Definition:
            PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)
            where P is current and Q is baseline distribution

        Interpretation (industry standard):
        - PSI < 0.10: No significant change
        - 0.10 ≤ PSI < 0.25: Moderate change, investigate
        - PSI ≥ 0.25: Significant change, action likely needed

        Args:
            current_distribution: Class -> probability mapping (current)
            baseline_distribution: Class -> probability mapping (baseline)
            epsilon: Small value for numerical stability (avoid log(0))

        Returns:
            PSI value (non-negative). Higher = more drift.

        Note:
            Missing classes in either distribution are assigned epsilon
            probability to avoid division by zero.

        Example:
            >>> baseline = {"A": 0.6, "B": 0.3, "C": 0.1}
            >>> current = {"A": 0.3, "B": 0.4, "C": 0.3}
            >>> psi = DriftDetector.population_stability_index(current, baseline)
        """
        # Handle empty distributions
        if not current_distribution or not baseline_distribution:
            logger.warning("Empty distribution provided to PSI calculation")
            return 0.0

        # Get all classes from both distributions
        all_classes = set(current_distribution.keys()) | set(baseline_distribution.keys())

        if not all_classes:
            return 0.0

        psi = 0.0
        for cls in all_classes:
            current_p = current_distribution.get(cls, epsilon)
            baseline_p = baseline_distribution.get(cls, epsilon)

            # Ensure positive values (clamp to epsilon)
            current_p = max(current_p, epsilon)
            baseline_p = max(baseline_p, epsilon)

            # PSI contribution for this class
            psi += (current_p - baseline_p) * np.log(current_p / baseline_p)

        # PSI should be non-negative (can be slightly negative due to floating point)
        return float(max(0.0, psi))

    # -------------------------------------------------------------------------
    # Embedding Space: Cosine Distance
    # -------------------------------------------------------------------------

    @staticmethod
    def cosine_distance(
        current_centroid: ArrayLike,
        baseline_centroid: ArrayLike,
    ) -> float:
        """
        Cosine distance between embedding centroids.

        Measures angular distance between two vectors, independent of
        magnitude. Useful for comparing semantic representations.

        Mathematical Definition:
            distance = 1 - cos(θ) = 1 - (A · B) / (||A|| × ||B||)

        Args:
            current_centroid: Mean embedding from current window
            baseline_centroid: Mean embedding from baseline period

        Returns:
            Cosine distance in [0, 2]:
            - 0.0: Identical direction (no drift)
            - 1.0: Orthogonal vectors
            - 2.0: Opposite direction (maximum drift)

        Note:
            Returns 1.0 (maximum uncertainty) if either vector has zero norm.

        Example:
            >>> baseline = [0.1, 0.2, 0.3]
            >>> current = [0.15, 0.25, 0.35]
            >>> dist = DriftDetector.cosine_distance(current, baseline)
        """
        current = np.asarray(current_centroid, dtype=np.float64).ravel()
        baseline = np.asarray(baseline_centroid, dtype=np.float64).ravel()

        # Handle dimension mismatch
        if len(current) != len(baseline):
            logger.warning(
                f"Dimension mismatch: current={len(current)}, baseline={len(baseline)}. "
                "Returning maximum distance."
            )
            return 1.0

        # Handle empty vectors
        if len(current) == 0:
            logger.warning("Empty vectors provided to cosine_distance")
            return 1.0

        # Compute norms
        norm_current = np.linalg.norm(current)
        norm_baseline = np.linalg.norm(baseline)

        # Handle zero vectors
        if norm_current < EPSILON or norm_baseline < EPSILON:
            logger.debug("Zero vector detected in cosine_distance")
            return 1.0

        # Compute cosine similarity
        cosine_similarity = np.dot(current, baseline) / (norm_current * norm_baseline)

        # Clamp to valid range (handles floating point errors)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return float(1.0 - cosine_similarity)

    @staticmethod
    def euclidean_distance(
        current_centroid: ArrayLike,
        baseline_centroid: ArrayLike,
    ) -> float:
        """
        Euclidean distance between embedding centroids.

        Standard L2 distance in embedding space. Unlike cosine distance,
        this is sensitive to vector magnitude.

        Args:
            current_centroid: Mean embedding from current window
            baseline_centroid: Mean embedding from baseline period

        Returns:
            Euclidean distance (non-negative)
        """
        current = np.asarray(current_centroid, dtype=np.float64).ravel()
        baseline = np.asarray(baseline_centroid, dtype=np.float64).ravel()

        if len(current) != len(baseline):
            logger.warning(
                f"Dimension mismatch: current={len(current)}, baseline={len(baseline)}"
            )
            return float('inf')

        return float(np.linalg.norm(current - baseline))

    # -------------------------------------------------------------------------
    # Uncertainty: Shannon Entropy
    # -------------------------------------------------------------------------

    @staticmethod
    def entropy(
        probabilities: ArrayLike,
        epsilon: float = EPSILON,
    ) -> float:
        """
        Shannon entropy of a probability distribution.

        Measures the uncertainty or information content of a distribution.
        Higher entropy = more uncertainty.

        Mathematical Definition:
            H(P) = -Σ p_i × log(p_i)

        Args:
            probabilities: Probability distribution (should sum to 1)
            epsilon: Small value to avoid log(0)

        Returns:
            Entropy value (non-negative). Units are nats (natural log).

        Example:
            >>> uniform = [0.25, 0.25, 0.25, 0.25]  # High entropy
            >>> certain = [0.99, 0.01, 0.0, 0.0]    # Low entropy
            >>> DriftDetector.entropy(uniform) > DriftDetector.entropy(certain)
            True
        """
        probs = np.asarray(probabilities, dtype=np.float64).ravel()

        # Handle empty input
        if probs.size == 0:
            return 0.0

        # Normalize if needed (silently, as this is common)
        prob_sum = np.sum(probs)
        if prob_sum > EPSILON:
            probs = probs / prob_sum
        else:
            return 0.0

        # Clamp to avoid log(0)
        probs = np.clip(probs, epsilon, 1.0)

        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def entropy_change(
        current_entropies: ArrayLike,
        baseline_entropy_mean: float,
    ) -> float:
        """
        Relative change in prediction entropy from baseline.

        Measures how much uncertainty has changed. Positive values
        indicate increased uncertainty (model less confident).

        Args:
            current_entropies: Entropy values from current window
            baseline_entropy_mean: Mean entropy from baseline period

        Returns:
            Relative change: (current_mean - baseline) / baseline
            - Positive: More uncertainty than baseline
            - Negative: Less uncertainty than baseline
            - 0.0: No change or invalid input
        """
        if baseline_entropy_mean <= EPSILON:
            logger.debug("Baseline entropy near zero, returning 0.0")
            return 0.0

        entropies = np.asarray(current_entropies, dtype=np.float64).ravel()

        if entropies.size == 0:
            return 0.0

        # Filter invalid values
        valid_entropies = entropies[~np.isnan(entropies) & ~np.isinf(entropies)]

        if valid_entropies.size == 0:
            return 0.0

        current_mean = np.mean(valid_entropies)
        return float((current_mean - baseline_entropy_mean) / baseline_entropy_mean)

    # -------------------------------------------------------------------------
    # Latency: Percentile Comparison
    # -------------------------------------------------------------------------

    @staticmethod
    def latency_drift(
        current_latencies: ArrayLike,
        baseline_p50: float,
        baseline_p99: float,
    ) -> tuple[float, float]:
        """
        Relative change in latency percentiles from baseline.

        Compares P50 (median) and P99 (tail) latencies. Positive values
        indicate latency regression (slower).

        Args:
            current_latencies: Latency values from current window
            baseline_p50: Baseline 50th percentile latency
            baseline_p99: Baseline 99th percentile latency

        Returns:
            Tuple of (p50_change, p99_change):
            - Positive: Slower than baseline (regression)
            - Negative: Faster than baseline (improvement)
            - 0.0: No change or invalid input
        """
        latencies = np.asarray(current_latencies, dtype=np.float64).ravel()

        if latencies.size < 2:
            logger.debug("Insufficient latency samples")
            return 0.0, 0.0

        # Remove invalid values
        valid_latencies = latencies[~np.isnan(latencies) & ~np.isinf(latencies)]
        valid_latencies = valid_latencies[valid_latencies >= 0]

        if valid_latencies.size < 2:
            return 0.0, 0.0

        current_p50 = float(np.percentile(valid_latencies, 50))
        current_p99 = float(np.percentile(valid_latencies, 99))

        # Calculate relative changes
        p50_change = 0.0 if baseline_p50 <= EPSILON else (current_p50 - baseline_p50) / baseline_p50
        p99_change = 0.0 if baseline_p99 <= EPSILON else (current_p99 - baseline_p99) / baseline_p99

        return float(p50_change), float(p99_change)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_prediction_distribution(
        predictions: list[str],
    ) -> dict[str, float]:
        """
        Compute probability distribution over prediction classes.

        Args:
            predictions: List of prediction values (as strings)

        Returns:
            Dictionary mapping class -> probability

        Example:
            >>> preds = ["A", "A", "B", "A", "C"]
            >>> dist = DriftDetector.compute_prediction_distribution(preds)
            >>> dist  # {"A": 0.6, "B": 0.2, "C": 0.2}
        """
        if not predictions:
            return {}

        # Count occurrences
        counts: dict[str, int] = {}
        for pred in predictions:
            key = str(pred)
            counts[key] = counts.get(key, 0) + 1

        # Normalize to probabilities
        total = len(predictions)
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def compute_centroid(
        embeddings: ArrayLike,
    ) -> list[float]:
        """
        Compute centroid (mean) of embedding vectors.

        Args:
            embeddings: 2D array of shape (n_samples, embedding_dim)

        Returns:
            Centroid vector as list of floats

        Example:
            >>> embs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            >>> centroid = DriftDetector.compute_centroid(embs)
            >>> centroid  # [0.667, 0.667]
        """
        arr = np.asarray(embeddings, dtype=np.float64)

        if arr.size == 0:
            return []

        # Handle 1D input (single embedding)
        if arr.ndim == 1:
            return cast("list[float]", arr.tolist())

        # Compute mean along sample axis
        centroid = np.mean(arr, axis=0)
        return cast("list[float]", centroid.tolist())

    @staticmethod
    def compute_percentiles(
        values: ArrayLike,
        percentiles: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Compute multiple percentiles of a distribution.

        Args:
            values: Numeric values to analyze
            percentiles: List of percentiles to compute (default: [50, 90, 95, 99])

        Returns:
            Dictionary mapping "p{N}" -> value
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]

        arr = np.asarray(values, dtype=np.float64).ravel()

        if arr.size == 0:
            return {f"p{p}": 0.0 for p in percentiles}

        valid = arr[~np.isnan(arr) & ~np.isinf(arr)]

        if valid.size == 0:
            return {f"p{p}": 0.0 for p in percentiles}

        result = {}
        for p in percentiles:
            result[f"p{p}"] = float(np.percentile(valid, p))

        return result
