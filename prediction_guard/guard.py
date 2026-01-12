"""
Prediction Guard - Main orchestration module.

This is the primary entry point for using Prediction Guard.
Brings together all components into a cohesive system.
"""

from datetime import datetime
from typing import Any

from .action import ActionExecutor
from .analysis import BaselineManager, OfflineAnalyzer
from .config import load_config
from .decision import DecisionEngine
from .incident import IncidentManager
from .types import (
    ActionType,
    AnalysisResult,
    GuardConfig,
    HealthDecision,
    ModelHealthState,
    RollbackAction,
)


class PredictionGuard:
    """
    Main orchestration class for Prediction Guard.

    Usage:
        guard = PredictionGuard()

        # Run analysis and get decision
        decision = guard.analyze_and_decide()

        # Execute action if warranted
        if decision.recommended_action == ActionType.ROLLBACK:
            guard.execute_action(decision)
    """

    def __init__(self, config: GuardConfig | None = None):
        """
        Initialize Prediction Guard.

        Args:
            config: Configuration (loads from file if not provided)
        """
        self.config = config or load_config()

        # Initialize components
        self.analyzer = OfflineAnalyzer(self.config)
        self.decision_engine = DecisionEngine(self.config)
        self.action_executor = ActionExecutor(self.config)
        self.incident_manager = IncidentManager(self.config)
        self.baseline_manager = BaselineManager(self.config.baseline_directory)

    def analyze(
        self,
        model_version: str | None = None,
        window_minutes: int | None = None,
    ) -> AnalysisResult | None:
        """
        Run drift analysis on recent predictions.

        Args:
            model_version: Version to analyze (defaults to config)
            window_minutes: Analysis window (defaults to config)

        Returns:
            AnalysisResult or None if insufficient data
        """
        return self.analyzer.analyze(
            model_version=model_version,
            window_minutes=window_minutes,
        )

    def decide(
        self,
        analysis: AnalysisResult,
        business_proxy_score: float | None = None,
    ) -> HealthDecision:
        """
        Make a health decision based on analysis.

        Args:
            analysis: Analysis result from analyze()
            business_proxy_score: Optional business metric (0=bad, 1=good)

        Returns:
            HealthDecision with state, reasons, and recommended action
        """
        return self.decision_engine.decide(analysis, business_proxy_score)

    def analyze_and_decide(
        self,
        model_version: str | None = None,
        business_proxy_score: float | None = None,
    ) -> HealthDecision | None:
        """
        Run full analysis and decision pipeline.

        Returns:
            HealthDecision or None if insufficient data
        """
        analysis = self.analyze(model_version=model_version)
        if analysis is None:
            return None

        return self.decide(analysis, business_proxy_score)

    def execute_action(
        self,
        decision: HealthDecision,
        force: bool = False,
    ) -> RollbackAction | None:
        """
        Execute recommended action if warranted.

        Args:
            decision: Health decision
            force: Bypass auto-rollback and cooldown checks

        Returns:
            RollbackAction if executed, None if skipped
        """
        return self.action_executor.execute(decision, force=force)

    def run_pipeline(
        self,
        model_version: str | None = None,
        business_proxy_score: float | None = None,
        auto_execute: bool = False,
    ) -> dict[str, Any]:
        """
        Run the complete pipeline: analyze -> decide -> (optionally) act.

        Args:
            model_version: Version to analyze
            business_proxy_score: Optional business metric
            auto_execute: If True, execute recommended action

        Returns:
            Pipeline result with analysis, decision, and action
        """
        result: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version or self.config.current_model_version,
            "analysis": None,
            "decision": None,
            "action": None,
            "incident_id": None,
        }

        # Analyze
        analysis = self.analyze(model_version=model_version)
        if analysis is None:
            result["error"] = "insufficient_data"
            return result

        result["analysis"] = {
            "sample_count": analysis.sample_count,
            "feature_drift_score": analysis.feature_drift_score,
            "embedding_drift_score": analysis.embedding_drift_score,
            "prediction_drift_score": analysis.prediction_drift_score,
            "confidence_entropy_change": analysis.confidence_entropy_change,
            "latency_p50_change": analysis.latency_p50_change,
            "latency_p99_change": analysis.latency_p99_change,
        }

        # Decide
        decision = self.decide(analysis, business_proxy_score)
        result["decision"] = decision.to_dict()

        # Record incident if not healthy
        if decision.state != ModelHealthState.HEALTHY:
            incident_id = self.incident_manager.record_incident(
                decision=decision,
                analysis=analysis,
            )
            result["incident_id"] = incident_id

        # Execute if requested and warranted
        if auto_execute and decision.recommended_action == ActionType.ROLLBACK:
            action = self.execute_action(decision)
            if action:
                result["action"] = action.to_dict()

                # Update incident with action
                if result["incident_id"]:
                    self.incident_manager.record_incident(
                        decision=decision,
                        analysis=analysis,
                        action=action,
                    )

        return result

    def create_baseline(
        self,
        model_version: str | None = None,
        days: int = 7,
    ) -> bool:
        """
        Create a baseline from historical data.

        Args:
            model_version: Version to create baseline for
            days: Days of data to use

        Returns:
            True if baseline was created successfully
        """
        return self.analyzer.create_baseline(model_version, days)

    def get_status(self) -> dict[str, Any]:
        """
        Get current system status.

        Returns information about:
        - Configuration
        - Available baselines
        - Recent incidents
        - Cooldown status
        """
        return {
            "current_model_version": self.config.current_model_version,
            "fallback_model_version": self.config.fallback_model_version,
            "auto_rollback_enabled": self.config.auto_rollback_enabled,
            "has_baseline": self.baseline_manager.has_baseline(self.config.current_model_version),
            "available_baselines": self.baseline_manager.list_baselines(),
            "recent_incidents": self.incident_manager.list_incidents(limit=5),
            "cooldown_active": self.action_executor.cooldown.is_cooling_down(),
            "cooldown_remaining_seconds": self.action_executor.cooldown.remaining_seconds(),
        }
