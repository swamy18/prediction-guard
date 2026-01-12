"""
Tests for the Decision Engine.
"""

import pytest
from datetime import datetime

from prediction_guard.types import (
    AnalysisResult,
    DriftMetric,
    DriftType,
    GuardConfig,
    ModelHealthState,
    ActionType,
)
from prediction_guard.decision.engine import DecisionEngine


def make_analysis(
    feature_drift: float = 0.0,
    embedding_drift: float = 0.0,
    prediction_drift: float = 0.0,
    entropy_change: float = 0.0,
    latency_change: float = 0.0,
    config: GuardConfig = None,
) -> AnalysisResult:
    """Helper to create analysis result with specified drift values."""
    config = config or GuardConfig()
    now = datetime.now()
    
    metrics = []
    
    # Feature drift
    metrics.append(DriftMetric(
        drift_type=DriftType.FEATURE,
        metric_name="ks_statistic",
        current_value=feature_drift,
        baseline_value=0.0,
        threshold=config.feature_drift_threshold,
        is_breached=feature_drift > config.feature_drift_threshold,
        window_start=now,
        window_end=now,
        sample_size=100,
    ))
    
    # Embedding drift
    metrics.append(DriftMetric(
        drift_type=DriftType.EMBEDDING,
        metric_name="cosine_distance",
        current_value=embedding_drift,
        baseline_value=0.0,
        threshold=config.embedding_drift_threshold,
        is_breached=embedding_drift > config.embedding_drift_threshold,
        window_start=now,
        window_end=now,
        sample_size=100,
    ))
    
    # Prediction drift
    metrics.append(DriftMetric(
        drift_type=DriftType.PREDICTION,
        metric_name="psi",
        current_value=prediction_drift,
        baseline_value=0.0,
        threshold=config.prediction_drift_threshold,
        is_breached=prediction_drift > config.prediction_drift_threshold,
        window_start=now,
        window_end=now,
        sample_size=100,
    ))
    
    # Confidence drift
    metrics.append(DriftMetric(
        drift_type=DriftType.CONFIDENCE,
        metric_name="entropy_change",
        current_value=entropy_change,
        baseline_value=0.0,
        threshold=config.confidence_entropy_threshold,
        is_breached=entropy_change > config.confidence_entropy_threshold,
        window_start=now,
        window_end=now,
        sample_size=100,
    ))
    
    return AnalysisResult(
        model_version="v1.0",
        analysis_timestamp=now,
        window_start=now,
        window_end=now,
        sample_count=100,
        drift_metrics=tuple(metrics),  # Convert to tuple for frozen dataclass
        feature_drift_score=feature_drift,
        embedding_drift_score=embedding_drift,
        prediction_drift_score=prediction_drift,
        confidence_entropy_change=entropy_change,
        latency_p50_change=latency_change,
        latency_p99_change=latency_change,
    )


class TestDecisionEngine:
    """Tests for the DecisionEngine class."""
    
    def test_healthy_no_drift(self):
        """No drift should result in healthy state."""
        config = GuardConfig()
        engine = DecisionEngine(config)
        
        analysis = make_analysis()
        decision = engine.decide(analysis)
        
        assert decision.state == ModelHealthState.HEALTHY
        assert decision.recommended_action == ActionType.NONE
        assert len(decision.reasons) == 0 or decision.reasons == ()
    
    def test_suspicious_single_drift(self):
        """Single drift signal should result in suspicious state."""
        config = GuardConfig()
        engine = DecisionEngine(config)
        
        # Only feature drift
        analysis = make_analysis(feature_drift=0.5, config=config)
        decision = engine.decide(analysis)
        
        assert decision.state == ModelHealthState.SUSPICIOUS
        assert decision.recommended_action == ActionType.ALERT
        assert "feature_drift_high" in decision.reasons
    
    def test_unstable_multiple_drift(self):
        """Multiple drift signals should result in unstable state."""
        config = GuardConfig()
        engine = DecisionEngine(config)
        
        # Three drift signals
        analysis = make_analysis(
            feature_drift=0.5,
            embedding_drift=0.5,
            prediction_drift=0.5,
            config=config,
        )
        decision = engine.decide(analysis)
        
        assert decision.state == ModelHealthState.UNSTABLE
        assert decision.recommended_action == ActionType.ROLLBACK
    
    def test_embedding_plus_confidence_is_unstable(self):
        """Embedding + confidence drift should trigger unstable."""
        config = GuardConfig()
        engine = DecisionEngine(config)
        
        analysis = make_analysis(
            embedding_drift=0.5,
            entropy_change=0.5,
            config=config,
        )
        decision = engine.decide(analysis)
        
        assert decision.state == ModelHealthState.UNSTABLE
    
    def test_business_proxy_healthy_overrides(self):
        """Healthy business metrics should override drift signals."""
        config = GuardConfig(business_proxy_overrides_drift=True)
        engine = DecisionEngine(config)
        
        # Severe drift
        analysis = make_analysis(
            feature_drift=0.5,
            embedding_drift=0.5,
            config=config,
        )
        
        # But business is healthy (score close to 1.0)
        decision = engine.decide(analysis, business_proxy_score=0.95)
        
        assert decision.state == ModelHealthState.HEALTHY
        assert "business_proxy_healthy_override" in decision.reasons
    
    def test_business_proxy_degraded_triggers_unstable(self):
        """Poor business metrics should trigger unstable."""
        config = GuardConfig(business_proxy_threshold=0.1)
        engine = DecisionEngine(config)
        
        # No drift
        analysis = make_analysis(config=config)
        
        # But business is degraded
        decision = engine.decide(analysis, business_proxy_score=0.05)
        
        assert decision.state == ModelHealthState.UNSTABLE
        assert "business_proxy_degraded" in decision.reasons
    
    def test_decision_includes_all_reasons(self):
        """Decision should include reasons for all breached thresholds."""
        config = GuardConfig()
        engine = DecisionEngine(config)
        
        analysis = make_analysis(
            feature_drift=0.5,
            embedding_drift=0.5,
            entropy_change=0.5,
            config=config,
        )
        decision = engine.decide(analysis)
        
        assert "feature_drift_high" in decision.reasons
        assert "embedding_drift_high" in decision.reasons
        assert "confidence_entropy_spike" in decision.reasons
    
    def test_explain_decision_readable(self):
        """Explanation should be human readable."""
        config = GuardConfig()
        engine = DecisionEngine(config)
        
        analysis = make_analysis(embedding_drift=0.5, config=config)
        decision = engine.decide(analysis)
        
        explanation = engine.explain_decision(decision)
        explanation_lower = explanation.lower()
        
        assert "model version" in explanation_lower
        assert "state" in explanation_lower
        assert "reasons" in explanation_lower
