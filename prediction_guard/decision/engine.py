"""
Decision Engine - The brain of Prediction Guard.

This module converts drift metrics into explicit health decisions
with human-readable reasons. It implements a multi-signal decision
logic that requires corroborating evidence before recommending action.

Design Principles:
1. Drift alone is NOT enough - multiple signals must align
2. Business proxy can override drift signals (reality trumps metrics)
3. Every decision has explicit, auditable reasons
4. Decision logic is readable by non-ML engineers

Decision Matrix:
    | Signals | State       | Action    |
    |---------|-------------|-----------|
    | 0       | HEALTHY     | None      |
    | 1       | SUSPICIOUS  | Alert     |
    | 2*      | UNSTABLE    | Rollback  |
    | 3+      | UNSTABLE    | Rollback  |

    *Embedding + Confidence triggers UNSTABLE even with only 2 signals

Author: Prediction Guard Team
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import datetime
from typing import Final

from ..types import (
    ActionType,
    AnalysisResult,
    DriftType,
    GuardConfig,
    HealthDecision,
    ModelHealthState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum signals for unstable state
MIN_SIGNALS_FOR_UNSTABLE: Final[int] = 3

# High-risk signal combinations (trigger unstable with only 2 signals)
HIGH_RISK_COMBINATIONS: Final[frozenset[frozenset[DriftType]]] = frozenset({
    frozenset({DriftType.EMBEDDING, DriftType.CONFIDENCE}),
    frozenset({DriftType.EMBEDDING, DriftType.PREDICTION}),
})

# Confidence levels for different decision scenarios
CONFIDENCE_NO_DRIFT: Final[float] = 0.95
CONFIDENCE_SINGLE_SIGNAL: Final[float] = 0.50
CONFIDENCE_TWO_SIGNALS: Final[float] = 0.65
CONFIDENCE_TWO_SIGNALS_HIGH_RISK: Final[float] = 0.75
CONFIDENCE_MULTI_SIGNAL: Final[float] = 0.85
CONFIDENCE_BUSINESS_OVERRIDE: Final[float] = 0.80
CONFIDENCE_BUSINESS_DEGRADED: Final[float] = 0.90


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True, slots=True)
class DriftSignalSummary:
    """
    Summary of detected drift signals for decision making.

    Attributes:
        has_feature_drift: True if feature drift threshold breached
        has_embedding_drift: True if embedding drift threshold breached
        has_prediction_drift: True if prediction drift threshold breached
        has_confidence_drift: True if confidence entropy threshold breached
        has_latency_drift: True if latency threshold breached
        total_signals: Count of breached thresholds
    """

    has_feature_drift: bool
    has_embedding_drift: bool
    has_prediction_drift: bool
    has_confidence_drift: bool
    has_latency_drift: bool

    @property
    def total_signals(self) -> int:
        """Total number of drift signals detected."""
        return sum([
            self.has_feature_drift,
            self.has_embedding_drift,
            self.has_prediction_drift,
            self.has_confidence_drift,
            self.has_latency_drift,
        ])

    @property
    def active_drift_types(self) -> frozenset[DriftType]:
        """Set of active drift types."""
        types: set[DriftType] = set()
        if self.has_feature_drift:
            types.add(DriftType.FEATURE)
        if self.has_embedding_drift:
            types.add(DriftType.EMBEDDING)
        if self.has_prediction_drift:
            types.add(DriftType.PREDICTION)
        if self.has_confidence_drift:
            types.add(DriftType.CONFIDENCE)
        if self.has_latency_drift:
            types.add(DriftType.LATENCY)
        return frozenset(types)

    def is_high_risk_combination(self) -> bool:
        """
        Check if active signals form a high-risk combination.

        Some signal combinations are particularly concerning and
        warrant immediate action even with only 2 signals.
        """
        active = self.active_drift_types
        return any(combo.issubset(active) for combo in HIGH_RISK_COMBINATIONS)


# =============================================================================
# DECISION ENGINE
# =============================================================================

class DecisionEngine:
    """
    Evaluates analysis results and produces health decisions.

    The decision engine implements a multi-signal approach:
    - Single signal → SUSPICIOUS (investigate, don't act)
    - Multiple signals → UNSTABLE (action recommended)
    - Business proxy can override drift signals

    This design minimizes false positives while catching real issues.

    Usage:
        >>> engine = DecisionEngine(config)
        >>> decision = engine.decide(analysis_result)
        >>> print(decision.state)
        ModelHealthState.HEALTHY
    """

    def __init__(self, config: GuardConfig) -> None:
        """
        Initialize the decision engine.

        Args:
            config: Guard configuration with thresholds
        """
        self._config = config
        logger.debug(
            f"DecisionEngine initialized with thresholds: "
            f"feature={config.feature_drift_threshold}, "
            f"embedding={config.embedding_drift_threshold}, "
            f"prediction={config.prediction_drift_threshold}"
        )

    @property
    def config(self) -> GuardConfig:
        """Read-only access to configuration."""
        return self._config

    def decide(
        self,
        analysis: AnalysisResult,
        business_proxy_score: float | None = None,
    ) -> HealthDecision:
        """
        Make a health decision based on analysis results.

        Args:
            analysis: Analysis result from OfflineAnalyzer
            business_proxy_score: Optional business metric [0.0=bad, 1.0=good]

        Returns:
            HealthDecision with state, reasons, and recommended action
        """
        # Extract drift signal summary
        signals = self._extract_signals(analysis)

        # Build reason list from active signals
        reasons = self._build_reasons(signals)

        # Evaluate state and action
        state, action, confidence = self._evaluate(
            signals=signals,
            business_proxy_score=business_proxy_score,
            reasons=reasons,
        )

        # Build analysis summary
        analysis_summary = self._build_analysis_summary(
            analysis=analysis,
            business_proxy_score=business_proxy_score,
        )

        decision = HealthDecision(
            model_version=analysis.model_version,
            state=state,
            reasons=tuple(reasons),
            recommended_action=action,
            confidence=confidence,
            analysis_summary=analysis_summary,
            decision_timestamp=datetime.now(),
        )

        logger.info(
            f"Decision: state={state.value}, action={action.value}, "
            f"confidence={confidence:.2f}, reasons={len(reasons)}"
        )

        return decision

    def _extract_signals(self, analysis: AnalysisResult) -> DriftSignalSummary:
        """
        Extract active drift signals from analysis result.

        Args:
            analysis: Analysis result with drift metrics

        Returns:
            Summary of which drift types are active
        """
        breached = analysis.breached_thresholds()

        return DriftSignalSummary(
            has_feature_drift=any(
                m.drift_type == DriftType.FEATURE for m in breached
            ),
            has_embedding_drift=any(
                m.drift_type == DriftType.EMBEDDING for m in breached
            ),
            has_prediction_drift=any(
                m.drift_type == DriftType.PREDICTION for m in breached
            ),
            has_confidence_drift=any(
                m.drift_type == DriftType.CONFIDENCE for m in breached
            ),
            has_latency_drift=any(
                m.drift_type == DriftType.LATENCY for m in breached
            ),
        )

    def _build_reasons(self, signals: DriftSignalSummary) -> list[str]:
        """
        Build human-readable reason strings from signals.

        Args:
            signals: Summary of active drift signals

        Returns:
            List of reason strings
        """
        reasons: list[str] = []

        if signals.has_feature_drift:
            reasons.append("feature_drift_high")
        if signals.has_embedding_drift:
            reasons.append("embedding_drift_high")
        if signals.has_prediction_drift:
            reasons.append("prediction_distribution_shift")
        if signals.has_confidence_drift:
            reasons.append("confidence_entropy_spike")
        if signals.has_latency_drift:
            reasons.append("latency_regression")

        return reasons

    def _evaluate(
        self,
        signals: DriftSignalSummary,
        business_proxy_score: float | None,
        reasons: list[str],
    ) -> tuple[ModelHealthState, ActionType, float]:
        """
        Core decision logic.

        This method implements the decision matrix with clear,
        readable rules that a non-ML engineer can understand.

        Args:
            signals: Summary of active drift signals
            business_proxy_score: Optional business metric
            reasons: Mutable list of reasons (may be appended to)

        Returns:
            Tuple of (state, action, confidence)
        """
        # === Business Proxy Override ===
        # If business metrics are available and enabled, they can override drift
        if business_proxy_score is not None and self._config.business_proxy_overrides_drift:

            # High business score = ignore drift (business is healthy)
            business_healthy_threshold = 1.0 - self._config.business_proxy_threshold
            if business_proxy_score >= business_healthy_threshold:
                if signals.total_signals > 0:
                    reasons.append("business_proxy_healthy_override")
                    logger.info(
                        f"Business proxy override: score={business_proxy_score:.2f} "
                        f"overrides {signals.total_signals} drift signals"
                    )
                return (
                    ModelHealthState.HEALTHY,
                    ActionType.NONE,
                    CONFIDENCE_BUSINESS_OVERRIDE,
                )

            # Low business score = critical, even without drift
            if business_proxy_score < self._config.business_proxy_threshold:
                reasons.append("business_proxy_degraded")
                logger.warning(
                    f"Business proxy degraded: score={business_proxy_score:.2f} "
                    f"below threshold {self._config.business_proxy_threshold}"
                )
                return (
                    ModelHealthState.UNSTABLE,
                    ActionType.ROLLBACK,
                    CONFIDENCE_BUSINESS_DEGRADED,
                )

        # === Multi-Signal Decision Logic ===
        signal_count = signals.total_signals

        # Case 1: No drift signals → HEALTHY
        if signal_count == 0:
            return (
                ModelHealthState.HEALTHY,
                ActionType.NONE,
                CONFIDENCE_NO_DRIFT,
            )

        # Case 2: Three or more signals → UNSTABLE (strong evidence)
        if signal_count >= MIN_SIGNALS_FOR_UNSTABLE:
            logger.warning(
                f"Multiple drift signals detected: {signal_count} signals. "
                f"Recommending rollback."
            )
            return (
                ModelHealthState.UNSTABLE,
                ActionType.ROLLBACK,
                CONFIDENCE_MULTI_SIGNAL,
            )

        # Case 3: Two signals
        if signal_count == 2:
            # Check for high-risk combinations
            if signals.is_high_risk_combination():
                logger.warning(
                    "High-risk drift combination detected "
                    "(embedding + confidence/prediction). Recommending rollback."
                )
                return (
                    ModelHealthState.UNSTABLE,
                    ActionType.ROLLBACK,
                    CONFIDENCE_TWO_SIGNALS_HIGH_RISK,
                )

            # Other two-signal combinations: suspicious
            return (
                ModelHealthState.SUSPICIOUS,
                ActionType.ALERT,
                CONFIDENCE_TWO_SIGNALS,
            )

        # Case 4: Single signal → SUSPICIOUS (could be noise)
        return (
            ModelHealthState.SUSPICIOUS,
            ActionType.ALERT,
            CONFIDENCE_SINGLE_SIGNAL,
        )

    def _build_analysis_summary(
        self,
        analysis: AnalysisResult,
        business_proxy_score: float | None,
    ) -> dict[str, float]:
        """
        Build summary of key metrics for the decision.

        Args:
            analysis: Analysis result
            business_proxy_score: Optional business metric

        Returns:
            Dictionary of metric name -> value
        """
        summary = {
            "feature_drift_score": analysis.feature_drift_score,
            "embedding_drift_score": analysis.embedding_drift_score,
            "prediction_drift_score": analysis.prediction_drift_score,
            "confidence_entropy_change": analysis.confidence_entropy_change,
            "latency_p50_change": analysis.latency_p50_change,
            "latency_p99_change": analysis.latency_p99_change,
        }

        if business_proxy_score is not None:
            summary["business_proxy_score"] = business_proxy_score

        return summary

    def explain_decision(self, decision: HealthDecision) -> str:
        """
        Generate a human-readable explanation of a decision.

        Useful for logging, alerting, and debugging.

        Args:
            decision: The health decision to explain

        Returns:
            Multi-line string explanation
        """
        lines = [
            "=" * 40,
            "MODEL HEALTH DECISION",
            "=" * 40,
            f"Model Version: {decision.model_version}",
            f"State:         {decision.state.value.upper()}",
            f"Confidence:    {decision.confidence:.0%}",
            f"Action:        {decision.recommended_action.value}",
            "",
            "REASONS:",
        ]

        if decision.reasons:
            for reason in decision.reasons:
                lines.append(f"  • {reason}")
        else:
            lines.append("  • No issues detected")

        lines.append("")
        lines.append("KEY METRICS:")
        for key, value in decision.analysis_summary.items():
            lines.append(f"  {key}: {value:.4f}")

        lines.append("=" * 40)

        return "\n".join(lines)

    def evaluate_threshold_breach(
        self,
        metric_value: float,
        threshold: float,
    ) -> bool:
        """
        Check if a metric value breaches a threshold.

        Simple utility for consistent threshold comparison.

        Args:
            metric_value: Current metric value
            threshold: Configured threshold

        Returns:
            True if metric exceeds threshold
        """
        return metric_value > threshold
