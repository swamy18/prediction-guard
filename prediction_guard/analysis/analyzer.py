"""
Offline Analyzer - Orchestrates drift analysis over logged events.

Reads recent logs, computes all drift metrics, and produces AnalysisResult.
This is meant to run on a schedule (cron) or manually.
"""

from datetime import datetime

from ..logging import LogReader
from ..types import AnalysisResult, DriftMetric, DriftType, GuardConfig
from .baseline_manager import BaselineManager
from .drift_detector import DriftDetector


class OfflineAnalyzer:
    """
    Runs comprehensive drift analysis over a window of prediction events.

    Usage:
        analyzer = OfflineAnalyzer(config)
        result = analyzer.analyze()
    """

    def __init__(self, config: GuardConfig):
        """
        Initialize the analyzer.

        Args:
            config: Guard configuration
        """
        self.config = config
        self.log_reader = LogReader(config.log_directory)
        self.baseline_manager = BaselineManager(config.baseline_directory)
        self.drift_detector = DriftDetector()

    def analyze(
        self,
        model_version: str | None = None,
        window_minutes: int | None = None,
    ) -> AnalysisResult | None:
        """
        Run full analysis on recent events.

        Args:
            model_version: Version to analyze (defaults to config)
            window_minutes: Analysis window (defaults to config)

        Returns:
            AnalysisResult or None if insufficient data
        """
        model_version = model_version or self.config.current_model_version
        window_minutes = window_minutes or self.config.analysis_window_minutes

        # Load baseline
        baseline = self.baseline_manager.load_baseline(model_version)
        if baseline is None:
            # No baseline = can't compute drift
            return None

        # Load recent events
        events = self.log_reader.read_window(
            minutes=window_minutes,
            model_version=model_version,
        )

        if len(events) < self.config.min_samples_for_analysis:
            # Insufficient data for analysis
            return None

        # Compute all drift metrics
        drift_metrics = []
        now = datetime.now()
        window_start = min(e.timestamp for e in events)
        window_end = max(e.timestamp for e in events)

        # --- Feature Drift (using confidence as proxy feature) ---
        confidence_scores = [e.confidence_score for e in events]
        ks_stat, _ks_pvalue = self.drift_detector.ks_test(
            confidence_scores,
            [baseline["confidence_mean"]] * 100  # Approximate baseline distribution
        )
        feature_drift_score = ks_stat

        drift_metrics.append(DriftMetric(
            drift_type=DriftType.FEATURE,
            metric_name="ks_statistic",
            current_value=ks_stat,
            baseline_value=0.0,
            threshold=self.config.feature_drift_threshold,
            is_breached=ks_stat > self.config.feature_drift_threshold,
            window_start=window_start,
            window_end=window_end,
            sample_size=len(events),
        ))

        # --- Embedding Drift ---
        current_centroid = self.drift_detector.compute_centroid(
            [e.embedding_summary for e in events]
        )
        baseline_centroid = baseline["embedding_centroid"]

        cosine_dist = self.drift_detector.cosine_distance(
            current_centroid, baseline_centroid
        )
        embedding_drift_score = cosine_dist

        drift_metrics.append(DriftMetric(
            drift_type=DriftType.EMBEDDING,
            metric_name="cosine_distance",
            current_value=cosine_dist,
            baseline_value=0.0,
            threshold=self.config.embedding_drift_threshold,
            is_breached=cosine_dist > self.config.embedding_drift_threshold,
            window_start=window_start,
            window_end=window_end,
            sample_size=len(events),
        ))

        # --- Prediction Drift ---
        current_distribution = self.drift_detector.compute_prediction_distribution(
            [str(e.prediction) for e in events]
        )
        baseline_distribution = baseline["prediction_distribution"]

        psi = self.drift_detector.population_stability_index(
            current_distribution, baseline_distribution
        )
        prediction_drift_score = psi

        drift_metrics.append(DriftMetric(
            drift_type=DriftType.PREDICTION,
            metric_name="psi",
            current_value=psi,
            baseline_value=0.0,
            threshold=self.config.prediction_drift_threshold,
            is_breached=psi > self.config.prediction_drift_threshold,
            window_start=window_start,
            window_end=window_end,
            sample_size=len(events),
        ))

        # --- Confidence Entropy Drift ---
        current_entropies = [e.prediction_entropy for e in events]
        entropy_change = self.drift_detector.entropy_change(
            current_entropies, baseline["confidence_entropy_mean"]
        )

        drift_metrics.append(DriftMetric(
            drift_type=DriftType.CONFIDENCE,
            metric_name="entropy_change",
            current_value=entropy_change,
            baseline_value=0.0,
            threshold=self.config.confidence_entropy_threshold,
            is_breached=entropy_change > self.config.confidence_entropy_threshold,
            window_start=window_start,
            window_end=window_end,
            sample_size=len(events),
        ))

        # --- Latency Drift ---
        current_latencies = [e.latency_ms for e in events]
        p50_change, p99_change = self.drift_detector.latency_drift(
            current_latencies,
            baseline["latency_p50"],
            baseline["latency_p99"],
        )

        drift_metrics.append(DriftMetric(
            drift_type=DriftType.LATENCY,
            metric_name="p99_latency_change",
            current_value=p99_change,
            baseline_value=0.0,
            threshold=self.config.latency_p99_threshold_ms / baseline["latency_p99"],
            is_breached=p99_change > (self.config.latency_p99_threshold_ms / baseline["latency_p99"]),
            window_start=window_start,
            window_end=window_end,
            sample_size=len(events),
        ))

        return AnalysisResult(
            model_version=model_version,
            analysis_timestamp=now,
            window_start=window_start,
            window_end=window_end,
            sample_count=len(events),
            drift_metrics=tuple(drift_metrics),  # Convert list to tuple
            feature_drift_score=feature_drift_score,
            embedding_drift_score=embedding_drift_score,
            prediction_drift_score=prediction_drift_score,
            confidence_entropy_change=entropy_change,
            latency_p50_change=p50_change,
            latency_p99_change=p99_change,
        )

    def create_baseline(
        self,
        model_version: str | None = None,
        days: int | None = None,
    ) -> bool:
        """
        Create a baseline from historical data.

        Args:
            model_version: Version to create baseline for
            days: Days of data to use

        Returns:
            True if baseline was created
        """
        model_version = model_version or self.config.current_model_version
        days = days or self.config.baseline_window_days

        events = self.log_reader.read_window(
            hours=days * 24,
            model_version=model_version,
        )

        if len(events) < self.config.min_samples_for_analysis:
            return False

        self.baseline_manager.compute_baseline_from_events(events, model_version)
        return True
