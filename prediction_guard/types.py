"""
Core type definitions for Prediction Guard.

This module defines all data structures used throughout the system.
Design principles:
- Immutable where possible (frozen dataclasses)
- Explicit validation at construction time
- Self-documenting with comprehensive docstrings
- Serialization/deserialization with explicit methods
- No magic strings - all states are enums

Author: Prediction Guard Team
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Final

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_EMBEDDING_DIM: Final[int] = 768
MAX_REASONS_LENGTH: Final[int] = 100
MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0


# =============================================================================
# ENUMS - Explicit states with no ambiguity
# =============================================================================

class ModelHealthState(str, Enum):
    """
    Enumeration of possible model health states.

    The system uses a three-state model for clarity:
    - HEALTHY: No action required
    - SUSPICIOUS: Alert but don't act automatically
    - UNSTABLE: Action recommended/required

    This design avoids the ambiguity of numeric scores while maintaining
    clear decision boundaries.
    """

    HEALTHY = "healthy"
    """Model is performing within expected parameters."""

    SUSPICIOUS = "suspicious"
    """Drift detected but not conclusive. Warrants investigation."""

    UNSTABLE = "unstable"
    """Clear degradation detected. Rollback recommended."""

    def __str__(self) -> str:
        return self.value

    @property
    def requires_action(self) -> bool:
        """Returns True if this state typically requires action."""
        return self == ModelHealthState.UNSTABLE

    @property
    def requires_investigation(self) -> bool:
        """Returns True if this state warrants human review."""
        return self in (ModelHealthState.SUSPICIOUS, ModelHealthState.UNSTABLE)


class DriftType(str, Enum):
    """
    Types of drift detected by the analysis system.

    Each drift type corresponds to a specific statistical test
    and has different implications for model health.
    """

    FEATURE = "feature_drift"
    """Input feature distribution has shifted (KS test)."""

    EMBEDDING = "embedding_drift"
    """Embedding space has shifted (cosine distance from centroid)."""

    PREDICTION = "prediction_drift"
    """Output prediction distribution has shifted (PSI)."""

    LATENCY = "latency_drift"
    """Response time has regressed (percentile comparison)."""

    CONFIDENCE = "confidence_drift"
    """Prediction certainty has changed (entropy analysis)."""

    def __str__(self) -> str:
        return self.value


class ActionType(str, Enum):
    """
    Actions that the system can recommend or execute.

    Ordered by severity: NONE < ALERT < PAUSE_TRAFFIC < ROLLBACK
    """

    NONE = "none"
    """No action required."""

    ALERT = "alert"
    """Send notification for human review."""

    PAUSE_TRAFFIC = "pause_traffic"
    """Temporarily reduce traffic to the model."""

    ROLLBACK = "rollback"
    """Revert to previous model version."""

    def __str__(self) -> str:
        return self.value

    @property
    def is_destructive(self) -> bool:
        """Returns True if this action modifies system state."""
        return self in (ActionType.PAUSE_TRAFFIC, ActionType.ROLLBACK)


class RollbackMechanism(str, Enum):
    """
    Supported mechanisms for executing model rollback.

    Each mechanism has different latency, reliability, and
    infrastructure requirements.
    """

    MODEL_ALIAS = "model_alias"
    """Update a model alias file that the inference service watches."""

    ENV_VAR = "env_var"
    """Update environment variable (requires service restart or hot-reload)."""

    CONFIG_FILE = "config_file"
    """Update configuration file (simplest, but may require polling)."""

    FEATURE_FLAG = "feature_flag"
    """Update feature flag (best for gradual rollout/rollback)."""

    def __str__(self) -> str:
        return self.value


# =============================================================================
# DATA CLASSES - Immutable, validated, serializable
# =============================================================================

@dataclass(frozen=True, slots=True)
class PredictionEvent:
    """
    A single prediction event logged by the middleware.

    This is the atomic unit of telemetry. Every prediction generates
    exactly one event. Events are immutable once created.

    Privacy guarantees:
    - `input_hash`: SHA256 hash of input (irreversible)
    - `embedding_summary`: Centroid only, not individual vectors
    - No PII is ever stored

    Attributes:
        timestamp: When the prediction was made (UTC recommended)
        model_version: Version identifier of the model used
        request_id: Unique identifier for this request (UUID)
        input_hash: SHA256 hash of the input data
        embedding_summary: Mean embedding vector (privacy-preserving)
        prediction: The model's output (class, score, etc.)
        confidence_score: Primary confidence metric [0, 1]
        prediction_entropy: Shannon entropy of prediction distribution
        latency_ms: End-to-end prediction latency in milliseconds
        request_context: Optional metadata (region, user_type, etc.)

    Example:
        >>> event = PredictionEvent(
        ...     timestamp=datetime.utcnow(),
        ...     model_version="v2.0",
        ...     request_id="550e8400-e29b-41d4-a716-446655440000",
        ...     input_hash="a3f2b8c9...",
        ...     embedding_summary=[0.1, 0.2, 0.3],
        ...     prediction="positive",
        ...     confidence_score=0.92,
        ...     prediction_entropy=0.28,
        ...     latency_ms=45.2,
        ...     request_context={"region": "us-east-1"}
        ... )
    """

    timestamp: datetime
    model_version: str
    request_id: str
    input_hash: str
    embedding_summary: tuple[float, ...]  # Immutable sequence
    prediction: Any
    confidence_score: float
    prediction_entropy: float
    latency_ms: float
    request_context: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        # Validate confidence score range
        if not (MIN_CONFIDENCE <= self.confidence_score <= MAX_CONFIDENCE):
            # Use object.__setattr__ because dataclass is frozen
            object.__setattr__(
                self,
                'confidence_score',
                max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, self.confidence_score))
            )
            logger.warning(
                f"Confidence score {self.confidence_score} clamped to valid range"
            )

        # Validate latency is positive
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary for JSONL logging.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "request_id": self.request_id,
            "input_hash": self.input_hash,
            "embedding_summary": list(self.embedding_summary),
            "prediction": self.prediction,
            "confidence_score": self.confidence_score,
            "prediction_entropy": self.prediction_entropy,
            "latency_ms": self.latency_ms,
            "request_context": self.request_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionEvent:
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary with event fields (e.g., from JSON).

        Returns:
            PredictionEvent instance.

        Raises:
            KeyError: If required field is missing.
            ValueError: If field value is invalid.
        """
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_version=data["model_version"],
            request_id=data["request_id"],
            input_hash=data["input_hash"],
            embedding_summary=tuple(data["embedding_summary"]),
            prediction=data["prediction"],
            confidence_score=float(data["confidence_score"]),
            prediction_entropy=float(data["prediction_entropy"]),
            latency_ms=float(data["latency_ms"]),
            request_context=data.get("request_context", {}),
        )


@dataclass(frozen=True, slots=True)
class DriftMetric:
    """
    A single drift measurement with full context.

    Each metric captures:
    - What type of drift was measured
    - Current vs baseline values
    - Whether the threshold was breached
    - Time window and sample size for statistical validity

    Attributes:
        drift_type: Category of drift (feature, embedding, etc.)
        metric_name: Specific metric name (e.g., "ks_statistic", "psi")
        current_value: Measured value in the current window
        baseline_value: Reference value from baseline period
        threshold: Configured threshold for this metric
        is_breached: True if current_value exceeds threshold
        window_start: Start of the analysis window
        window_end: End of the analysis window
        sample_size: Number of samples used in calculation
    """

    drift_type: DriftType
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    is_breached: bool
    window_start: datetime
    window_end: datetime
    sample_size: int

    def __post_init__(self) -> None:
        """Validate metric values."""
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be non-negative, got {self.sample_size}")
        if self.threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")

    @property
    def severity(self) -> float:
        """
        Calculate how far over threshold (0.0 if not breached).

        Returns:
            Relative severity as (current - threshold) / threshold.
            Returns 0.0 if threshold is not breached.
        """
        if not self.is_breached or self.threshold == 0:
            return 0.0
        return (self.current_value - self.threshold) / self.threshold

    @property
    def window_duration_seconds(self) -> float:
        """Duration of the analysis window in seconds."""
        return (self.window_end - self.window_start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "drift_type": self.drift_type.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "threshold": self.threshold,
            "is_breached": self.is_breached,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sample_size": self.sample_size,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """
    Complete analysis result from the offline analyzer.

    Aggregates all drift metrics and provides summary scores
    for each drift category.

    Attributes:
        model_version: Version of the model analyzed
        analysis_timestamp: When the analysis was performed
        window_start: Start of the data window analyzed
        window_end: End of the data window analyzed
        sample_count: Total prediction events analyzed
        drift_metrics: List of individual drift measurements
        feature_drift_score: Aggregate feature drift (0.0-1.0)
        embedding_drift_score: Aggregate embedding drift (0.0-1.0)
        prediction_drift_score: Aggregate prediction drift (0.0-1.0)
        confidence_entropy_change: Relative change from baseline
        latency_p50_change: Relative change in P50 latency
        latency_p99_change: Relative change in P99 latency
        business_proxy_score: Optional business health metric
    """

    model_version: str
    analysis_timestamp: datetime
    window_start: datetime
    window_end: datetime
    sample_count: int
    drift_metrics: tuple[DriftMetric, ...]  # Immutable sequence
    feature_drift_score: float
    embedding_drift_score: float
    prediction_drift_score: float
    confidence_entropy_change: float
    latency_p50_change: float
    latency_p99_change: float
    business_proxy_score: float | None = None

    def __post_init__(self) -> None:
        """Validate analysis result."""
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be non-negative, got {self.sample_count}")

    def breached_thresholds(self) -> list[DriftMetric]:
        """
        Return only metrics that breached their thresholds.

        Returns:
            List of DriftMetric instances where is_breached is True.
        """
        return [m for m in self.drift_metrics if m.is_breached]

    @property
    def breach_count(self) -> int:
        """Number of metrics that breached their thresholds."""
        return sum(1 for m in self.drift_metrics if m.is_breached)

    @property
    def has_significant_drift(self) -> bool:
        """True if any metric breached its threshold."""
        return self.breach_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_version": self.model_version,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sample_count": self.sample_count,
            "drift_metrics": [m.to_dict() for m in self.drift_metrics],
            "feature_drift_score": self.feature_drift_score,
            "embedding_drift_score": self.embedding_drift_score,
            "prediction_drift_score": self.prediction_drift_score,
            "confidence_entropy_change": self.confidence_entropy_change,
            "latency_p50_change": self.latency_p50_change,
            "latency_p99_change": self.latency_p99_change,
            "business_proxy_score": self.business_proxy_score,
            "breach_count": self.breach_count,
        }


@dataclass(frozen=True, slots=True)
class HealthDecision:
    """
    The output of the decision engine.

    This is the core artifact of Prediction Guard - an explicit
    health state with human-readable reasons and a recommended action.

    Design principles:
    - State is always one of three explicit values
    - Reasons are human-readable strings
    - Confidence indicates decision certainty
    - Fully serializable for logging and APIs

    Attributes:
        model_version: Version this decision applies to
        state: One of HEALTHY, SUSPICIOUS, UNSTABLE
        reasons: List of human-readable reason strings
        recommended_action: What action to take
        confidence: Decision confidence (0.0 to 1.0)
        analysis_summary: Key metrics that drove the decision
        decision_timestamp: When the decision was made
    """

    model_version: str
    state: ModelHealthState
    reasons: tuple[str, ...]  # Immutable sequence
    recommended_action: ActionType
    confidence: float
    analysis_summary: dict[str, float]
    decision_timestamp: datetime

    def __post_init__(self) -> None:
        """Validate decision fields."""
        if not (MIN_CONFIDENCE <= self.confidence <= MAX_CONFIDENCE):
            raise ValueError(
                f"confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}, "
                f"got {self.confidence}"
            )

    @property
    def requires_action(self) -> bool:
        """True if the recommended action is destructive."""
        return self.recommended_action.is_destructive

    @property
    def reason_count(self) -> int:
        """Number of reasons for this decision."""
        return len(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize for logging and API responses.

        Returns:
            Dictionary with all decision fields.
        """
        return {
            "model_version": self.model_version,
            "state": self.state.value,
            "reasons": list(self.reasons),
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
            "analysis_summary": self.analysis_summary,
            "decision_timestamp": self.decision_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthDecision:
        """Deserialize from dictionary."""
        return cls(
            model_version=data["model_version"],
            state=ModelHealthState(data["state"]),
            reasons=tuple(data["reasons"]),
            recommended_action=ActionType(data["recommended_action"]),
            confidence=float(data["confidence"]),
            analysis_summary=data["analysis_summary"],
            decision_timestamp=datetime.fromisoformat(data["decision_timestamp"]),
        )


@dataclass(slots=True)
class RollbackAction:
    """
    Record of a rollback action for audit trail.

    Mutable because success/completion status is updated after execution.

    Attributes:
        action_id: Unique identifier for this action (UUID)
        from_version: Model version being rolled back from
        to_version: Model version being rolled back to
        mechanism: How the rollback was executed
        reason: Human-readable reason for rollback
        initiated_at: When the rollback was started
        completed_at: When the rollback finished (None if in progress)
        success: Whether the rollback succeeded
        error_message: Error details if rollback failed
        is_auto: True if automatically triggered (vs manual)
    """

    action_id: str
    from_version: str
    to_version: str
    mechanism: RollbackMechanism
    reason: str
    initiated_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    error_message: str | None = None
    is_auto: bool = False

    @property
    def is_complete(self) -> bool:
        """True if the action has finished (success or failure)."""
        return self.completed_at is not None

    @property
    def duration_seconds(self) -> float | None:
        """Duration of the rollback in seconds, or None if incomplete."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.initiated_at).total_seconds()

    def mark_complete(self, success: bool, error_message: str | None = None) -> None:
        """
        Mark this action as complete.

        Args:
            success: Whether the rollback succeeded
            error_message: Error details if failed
        """
        self.completed_at = datetime.now()
        self.success = success
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging."""
        return {
            "action_id": self.action_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "mechanism": self.mechanism.value,
            "reason": self.reason,
            "initiated_at": self.initiated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "error_message": self.error_message,
            "is_auto": self.is_auto,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class IncidentSnapshot:
    """
    Complete snapshot of an incident for post-mortem analysis.

    Used by the learning loop to capture context around
    incidents for threshold tuning.

    Attributes:
        incident_id: Unique identifier for this incident
        model_version: Model version involved
        detected_at: When the incident was detected
        decision: The health decision that triggered the incident
        analysis: The analysis result at the time
        action_taken: Rollback action if one was executed
        resolution_notes: Human notes after investigation
        threshold_adjustments: Suggested threshold changes
    """

    incident_id: str
    model_version: str
    detected_at: datetime
    decision: HealthDecision | dict[str, Any]
    analysis: AnalysisResult | dict[str, Any]
    action_taken: RollbackAction | dict[str, Any] | None = None
    resolution_notes: str | None = None
    threshold_adjustments: dict[str, float] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """True if resolution notes have been added."""
        return self.resolution_notes is not None

    @property
    def has_action(self) -> bool:
        """True if an action was taken."""
        return self.action_taken is not None


@dataclass
class GuardConfig:
    """
    Configuration for Prediction Guard.

    All thresholds and settings are explicit and tunable.
    Defaults are conservative (minimize false positives).

    Configuration can be loaded from:
    - Python code (direct instantiation)
    - JSON file (via config.load_config)
    - Environment variable pointing to JSON file

    Threshold Guidelines:
    - feature_drift_threshold: 0.10-0.20 typical
    - embedding_drift_threshold: 0.15-0.25 typical
    - prediction_drift_threshold: 0.05-0.15 typical
    - confidence_entropy_threshold: 0.20-0.35 typical

    Attributes:
        feature_drift_threshold: KS statistic threshold
        embedding_drift_threshold: Cosine distance threshold
        prediction_drift_threshold: PSI threshold
        confidence_entropy_threshold: Relative entropy change threshold
        latency_p99_threshold_ms: Absolute P99 latency threshold
        analysis_window_minutes: Size of analysis window
        baseline_window_days: Days of data for baseline calculation
        min_samples_for_analysis: Minimum events for valid analysis
        auto_rollback_enabled: Whether to auto-execute rollbacks
        rollback_cooldown_minutes: Minimum time between rollbacks
        rollback_mechanism: How to execute rollbacks
        log_directory: Where to store prediction logs
        baseline_directory: Where to store baselines
        incident_directory: Where to store incidents
        current_model_version: Version currently in production
        fallback_model_version: Version to roll back to
        business_proxy_enabled: Whether to use business metrics
        business_proxy_threshold: Business metric alert threshold
        business_proxy_overrides_drift: Let business trump drift signals
    """

    # Class constants for validation
    MIN_THRESHOLD: ClassVar[float] = 0.0
    MAX_THRESHOLD: ClassVar[float] = 1.0
    MIN_WINDOW_MINUTES: ClassVar[int] = 5
    MAX_WINDOW_MINUTES: ClassVar[int] = 1440  # 24 hours
    MIN_SAMPLES: ClassVar[int] = 10

    # Drift thresholds
    feature_drift_threshold: float = 0.15
    embedding_drift_threshold: float = 0.20
    prediction_drift_threshold: float = 0.10
    confidence_entropy_threshold: float = 0.25
    latency_p99_threshold_ms: float = 100.0

    # Analysis windows
    analysis_window_minutes: int = 60
    baseline_window_days: int = 7
    min_samples_for_analysis: int = 100

    # Rollback settings
    auto_rollback_enabled: bool = False  # CRITICAL: Off by default
    rollback_cooldown_minutes: int = 30
    rollback_mechanism: RollbackMechanism = field(
        default=RollbackMechanism.CONFIG_FILE
    )

    # Paths
    log_directory: str = "./logs"
    baseline_directory: str = "./baselines"
    incident_directory: str = "./incidents"

    # Model versions
    current_model_version: str = "v1"
    fallback_model_version: str = "v0"

    # Business proxy (optional)
    business_proxy_enabled: bool = False
    business_proxy_threshold: float = 0.10
    business_proxy_overrides_drift: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_thresholds()
        self._validate_windows()
        self._validate_versions()

    def _validate_thresholds(self) -> None:
        """Validate threshold values are in reasonable ranges."""
        thresholds = [
            ("feature_drift_threshold", self.feature_drift_threshold),
            ("embedding_drift_threshold", self.embedding_drift_threshold),
            ("prediction_drift_threshold", self.prediction_drift_threshold),
            ("confidence_entropy_threshold", self.confidence_entropy_threshold),
            ("business_proxy_threshold", self.business_proxy_threshold),
        ]

        for name, value in thresholds:
            if not (self.MIN_THRESHOLD <= value <= self.MAX_THRESHOLD):
                raise ValueError(
                    f"{name} must be between {self.MIN_THRESHOLD} and "
                    f"{self.MAX_THRESHOLD}, got {value}"
                )

        if self.latency_p99_threshold_ms <= 0:
            raise ValueError(
                f"latency_p99_threshold_ms must be positive, "
                f"got {self.latency_p99_threshold_ms}"
            )

    def _validate_windows(self) -> None:
        """Validate window and sample settings."""
        if not (self.MIN_WINDOW_MINUTES <= self.analysis_window_minutes <= self.MAX_WINDOW_MINUTES):
            raise ValueError(
                f"analysis_window_minutes must be between {self.MIN_WINDOW_MINUTES} "
                f"and {self.MAX_WINDOW_MINUTES}, got {self.analysis_window_minutes}"
            )

        if self.baseline_window_days < 1:
            raise ValueError(
                f"baseline_window_days must be at least 1, got {self.baseline_window_days}"
            )

        if self.min_samples_for_analysis < self.MIN_SAMPLES:
            raise ValueError(
                f"min_samples_for_analysis must be at least {self.MIN_SAMPLES}, "
                f"got {self.min_samples_for_analysis}"
            )

    def _validate_versions(self) -> None:
        """Validate model version settings."""
        if not self.current_model_version:
            raise ValueError("current_model_version cannot be empty")

        if not self.fallback_model_version:
            raise ValueError("fallback_model_version cannot be empty")

        if self.current_model_version == self.fallback_model_version:
            logger.warning(
                "current_model_version and fallback_model_version are the same. "
                "Rollback will have no effect."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "feature_drift_threshold": self.feature_drift_threshold,
            "embedding_drift_threshold": self.embedding_drift_threshold,
            "prediction_drift_threshold": self.prediction_drift_threshold,
            "confidence_entropy_threshold": self.confidence_entropy_threshold,
            "latency_p99_threshold_ms": self.latency_p99_threshold_ms,
            "analysis_window_minutes": self.analysis_window_minutes,
            "baseline_window_days": self.baseline_window_days,
            "min_samples_for_analysis": self.min_samples_for_analysis,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "rollback_cooldown_minutes": self.rollback_cooldown_minutes,
            "rollback_mechanism": self.rollback_mechanism.value,
            "log_directory": self.log_directory,
            "baseline_directory": self.baseline_directory,
            "incident_directory": self.incident_directory,
            "current_model_version": self.current_model_version,
            "fallback_model_version": self.fallback_model_version,
            "business_proxy_enabled": self.business_proxy_enabled,
            "business_proxy_threshold": self.business_proxy_threshold,
            "business_proxy_overrides_drift": self.business_proxy_overrides_drift,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuardConfig:
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with configuration fields.

        Returns:
            GuardConfig instance.
        """
        # Handle enum conversion
        if "rollback_mechanism" in data and isinstance(data["rollback_mechanism"], str):
            data = data.copy()
            data["rollback_mechanism"] = RollbackMechanism(data["rollback_mechanism"])

        return cls(**data)
