"""
Baseline Manager - Stores and retrieves baseline statistics for drift comparison.

Baselines capture "normal" behavior for a model version:
- Feature distributions
- Embedding centroids
- Prediction class distributions
- Latency percentiles
- Confidence distributions
"""

import json

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np


class BaselineManager:
    """
    Manages baseline statistics for model versions.

    Baselines are stored as JSON files, one per model version.
    """

    def __init__(self, baseline_directory: str):
        """
        Initialize the baseline manager.

        Args:
            baseline_directory: Directory for baseline files
        """
        self.baseline_directory = Path(baseline_directory)
        self.baseline_directory.mkdir(parents=True, exist_ok=True)

    def save_baseline(
        self,
        model_version: str,
        embedding_centroid: list[float],
        prediction_distribution: dict[str, float],
        confidence_mean: float,
        confidence_std: float,
        confidence_entropy_mean: float,
        latency_p50: float,
        latency_p99: float,
        sample_count: int,
        computed_from: datetime,
        computed_to: datetime,
    ) -> None:
        """
        Save a baseline for a model version.

        Args:
            model_version: Version identifier
            embedding_centroid: Mean embedding vector
            prediction_distribution: Class -> probability mapping
            confidence_mean: Mean confidence score
            confidence_std: Standard deviation of confidence
            confidence_entropy_mean: Mean prediction entropy
            latency_p50: 50th percentile latency (ms)
            latency_p99: 99th percentile latency (ms)
            sample_count: Number of samples used
            computed_from: Start of baseline window
            computed_to: End of baseline window
        """
        baseline = {
            "model_version": model_version,
            "embedding_centroid": embedding_centroid,
            "prediction_distribution": prediction_distribution,
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "confidence_entropy_mean": confidence_entropy_mean,
            "latency_p50": latency_p50,
            "latency_p99": latency_p99,
            "sample_count": sample_count,
            "computed_from": computed_from.isoformat(),
            "computed_to": computed_to.isoformat(),
            "saved_at": datetime.now().isoformat(),
        }

        filepath = self._baseline_path(model_version)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)

    def load_baseline(self, model_version: str) -> dict[str, Any] | None:
        """
        Load baseline for a model version.

        Returns:
            Baseline dictionary or None if not found
        """
        filepath = self._baseline_path(model_version)
        if not filepath.exists():
            return None

        with open(filepath, encoding="utf-8") as f:
            return cast("dict[str, Any]", json.load(f))

    def has_baseline(self, model_version: str) -> bool:
        """Check if a baseline exists for a model version."""
        return self._baseline_path(model_version).exists()

    def list_baselines(self) -> list[str]:
        """List all model versions with baselines."""
        versions = []
        for f in self.baseline_directory.glob("baseline_*.json"):
            version = f.stem.replace("baseline_", "")
            versions.append(version)
        return sorted(versions)

    def delete_baseline(self, model_version: str) -> bool:
        """
        Delete a baseline.

        Returns:
            True if deleted, False if not found
        """
        filepath = self._baseline_path(model_version)
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def _baseline_path(self, model_version: str) -> Path:
        """Get the filepath for a model version's baseline."""
        # Sanitize version for filename
        safe_version = model_version.replace("/", "_").replace("\\", "_")
        return self.baseline_directory / f"baseline_{safe_version}.json"

    def compute_baseline_from_events(
        self,
        events: list[Any],  # List[PredictionEvent]
        model_version: str,
    ) -> None:
        """
        Compute and save a baseline from a list of events.

        Args:
            events: List of PredictionEvent objects
            model_version: Version to save baseline for
        """
        if not events:
            raise ValueError("Cannot compute baseline from empty events list")

        # Compute embedding centroid
        embeddings = np.array([e.embedding_summary for e in events])
        embedding_centroid = cast("list[float]", np.mean(embeddings, axis=0).tolist())

        # Compute prediction distribution
        predictions = [str(e.prediction) for e in events]
        unique, counts = np.unique(predictions, return_counts=True)
        prediction_distribution = {
            str(k): float(v / len(predictions))
            for k, v in zip(unique, counts)
        }

        # Compute confidence stats
        confidences = [e.confidence_score for e in events]
        confidence_mean = float(np.mean(confidences))
        confidence_std = float(np.std(confidences))

        # Compute entropy stats
        entropies = [e.prediction_entropy for e in events]
        confidence_entropy_mean = float(np.mean(entropies))

        # Compute latency percentiles
        latencies = [e.latency_ms for e in events]
        latency_p50 = float(np.percentile(latencies, 50))
        latency_p99 = float(np.percentile(latencies, 99))

        # Get time bounds
        timestamps = [e.timestamp for e in events]
        computed_from = min(timestamps)
        computed_to = max(timestamps)

        self.save_baseline(
            model_version=model_version,
            embedding_centroid=embedding_centroid,
            prediction_distribution=prediction_distribution,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            confidence_entropy_mean=confidence_entropy_mean,
            latency_p50=latency_p50,
            latency_p99=latency_p99,
            sample_count=len(events),
            computed_from=computed_from,
            computed_to=computed_to,
        )
