"""
Incident Manager - Learning loop for Prediction Guard.

Saves incident snapshots for post-mortem analysis and threshold tuning.
No auto-learning in v1 - human-in-the-loop only.
"""

import json
import uuid

from datetime import datetime
from pathlib import Path
from typing import Any

from .types import (
    AnalysisResult,
    GuardConfig,
    HealthDecision,
    IncidentSnapshot,
    RollbackAction,
)


class IncidentManager:
    """
    Manages incident snapshots for learning and tuning.

    Each incident captures:
    - Decision context
    - Analysis metrics
    - Action taken
    - Human notes (for tuning)
    """

    def __init__(self, config: GuardConfig):
        self.config = config
        self.incident_dir = Path(config.incident_directory)
        self.incident_dir.mkdir(parents=True, exist_ok=True)

    def record_incident(
        self,
        decision: HealthDecision,
        analysis: AnalysisResult,
        action: RollbackAction | None = None,
    ) -> str:
        """
        Record an incident for later analysis.

        Returns:
            incident_id
        """
        incident_id = str(uuid.uuid4())

        snapshot = IncidentSnapshot(
            incident_id=incident_id,
            model_version=decision.model_version,
            detected_at=datetime.now(),
            decision=decision,
            analysis=analysis,
            action_taken=action,
        )

        self._save_snapshot(snapshot)
        return incident_id

    def add_resolution_notes(
        self,
        incident_id: str,
        notes: str,
        threshold_adjustments: dict[str, float] | None = None,
    ) -> bool:
        """
        Add human notes to an incident (for learning loop).

        Args:
            incident_id: ID of the incident
            notes: Human-written resolution notes
            threshold_adjustments: Suggested threshold changes

        Returns:
            True if updated successfully
        """
        snapshot = self._load_snapshot(incident_id)
        if snapshot is None:
            return False

        snapshot.resolution_notes = notes
        if threshold_adjustments:
            snapshot.threshold_adjustments = threshold_adjustments

        self._save_snapshot(snapshot)
        return True

    def list_incidents(
        self,
        model_version: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List recent incidents.

        Returns:
            List of incident summaries
        """
        incidents: list[dict[str, Any]] = []
        for f in sorted(self.incident_dir.glob("incident_*.json"), reverse=True):
            if len(incidents) >= limit:
                break

            try:
                with open(f) as file:
                    data = json.load(file)

                if model_version and data.get("model_version") != model_version:
                    continue

                incidents.append({
                    "incident_id": data["incident_id"],
                    "model_version": data["model_version"],
                    "detected_at": data["detected_at"],
                    "state": data["decision"]["state"],
                    "has_action": data["action_taken"] is not None,
                    "has_notes": bool(data.get("resolution_notes")),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return incidents

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        """Get full incident details."""
        snapshot = self._load_snapshot(incident_id)
        if snapshot is None:
            return None
        return self._snapshot_to_dict(snapshot)

    def get_threshold_recommendations(
        self,
        model_version: str | None = None,
    ) -> dict[str, float]:
        """
        Aggregate threshold adjustments from past incidents.

        Returns average of all suggested adjustments from resolved incidents.
        """
        adjustments: dict[str, list[float]] = {}

        for f in self.incident_dir.glob("incident_*.json"):
            try:
                with open(f) as file:
                    data = json.load(file)

                if model_version and data.get("model_version") != model_version:
                    continue

                if data.get("threshold_adjustments"):
                    for key, value in data["threshold_adjustments"].items():
                        if key not in adjustments:
                            adjustments[key] = []
                        adjustments[key].append(value)
            except (json.JSONDecodeError, KeyError):
                continue

        # Average the recommendations
        return {k: sum(v) / len(v) for k, v in adjustments.items()}

    def _save_snapshot(self, snapshot: IncidentSnapshot) -> None:
        filepath = self.incident_dir / f"incident_{snapshot.incident_id}.json"
        with open(filepath, "w") as f:
            json.dump(self._snapshot_to_dict(snapshot), f, indent=2)

    def _load_snapshot(self, incident_id: str) -> IncidentSnapshot | None:
        filepath = self.incident_dir / f"incident_{incident_id}.json"
        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        # Simplified reconstruction (full deserialization would need more work)
        return IncidentSnapshot(
            incident_id=data["incident_id"],
            model_version=data["model_version"],
            detected_at=datetime.fromisoformat(data["detected_at"]),
            decision=data["decision"],  # Keep as dict for simplicity
            analysis=data["analysis"],  # Keep as dict for simplicity
            action_taken=data.get("action_taken"),
            resolution_notes=data.get("resolution_notes"),
            threshold_adjustments=data.get("threshold_adjustments", {}),
        )

    def _snapshot_to_dict(self, snapshot: IncidentSnapshot) -> dict[str, Any]:
        return {
            "incident_id": snapshot.incident_id,
            "model_version": snapshot.model_version,
            "detected_at": snapshot.detected_at.isoformat(),
            "decision": snapshot.decision.to_dict() if hasattr(snapshot.decision, 'to_dict') else snapshot.decision,
            "analysis": self._analysis_to_dict(snapshot.analysis),
            "action_taken": snapshot.action_taken.to_dict() if snapshot.action_taken and hasattr(snapshot.action_taken, 'to_dict') else snapshot.action_taken,
            "resolution_notes": snapshot.resolution_notes,
            "threshold_adjustments": snapshot.threshold_adjustments,
        }

    def _analysis_to_dict(self, analysis: AnalysisResult | dict[str, Any]) -> dict[str, Any]:
        if isinstance(analysis, dict):
            return analysis
        return {
            "model_version": analysis.model_version,
            "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
            "sample_count": analysis.sample_count,
            "feature_drift_score": analysis.feature_drift_score,
            "embedding_drift_score": analysis.embedding_drift_score,
            "prediction_drift_score": analysis.prediction_drift_score,
            "confidence_entropy_change": analysis.confidence_entropy_change,
            "latency_p50_change": analysis.latency_p50_change,
            "latency_p99_change": analysis.latency_p99_change,
        }
