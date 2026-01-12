"""
Tests for Prediction Guard drift detection.
"""

import pytest
import numpy as np
from prediction_guard.analysis.drift_detector import DriftDetector


class TestDriftDetector:
    """Tests for the DriftDetector class."""
    
    def test_ks_test_identical_distributions(self):
        """KS test should show no drift for identical distributions."""
        values = [0.5] * 100
        stat, pvalue = DriftDetector.ks_test(values, values)
        assert stat == 0.0
        assert pvalue == 1.0
    
    def test_ks_test_different_distributions(self):
        """KS test should detect drift between different distributions."""
        baseline = list(np.random.normal(0, 1, 100))
        current = list(np.random.normal(2, 1, 100))  # Shifted mean
        
        stat, pvalue = DriftDetector.ks_test(current, baseline)
        assert stat > 0.5  # Significant shift
        assert pvalue < 0.05  # Low p-value
    
    def test_psi_identical_distributions(self):
        """PSI should be 0 for identical distributions."""
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}
        psi = DriftDetector.population_stability_index(dist, dist)
        assert abs(psi) < 0.001
    
    def test_psi_shifted_distribution(self):
        """PSI should detect categorical shift."""
        baseline = {"a": 0.6, "b": 0.3, "c": 0.1}
        current = {"a": 0.2, "b": 0.3, "c": 0.5}  # Shifted
        
        psi = DriftDetector.population_stability_index(current, baseline)
        assert psi > 0.25  # Significant shift threshold
    
    def test_cosine_distance_identical_vectors(self):
        """Cosine distance should be 0 for identical vectors."""
        vec = [0.5, 0.5, 0.5]
        dist = DriftDetector.cosine_distance(vec, vec)
        assert abs(dist) < 0.001
    
    def test_cosine_distance_orthogonal_vectors(self):
        """Cosine distance should be 1 for orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        dist = DriftDetector.cosine_distance(vec1, vec2)
        assert abs(dist - 1.0) < 0.001
    
    def test_entropy_uniform_distribution(self):
        """Entropy should be maximum for uniform distribution."""
        # Uniform over 4 classes
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = DriftDetector.entropy(probs)
        max_entropy = np.log(4)
        assert abs(entropy - max_entropy) < 0.001
    
    def test_entropy_certain_prediction(self):
        """Entropy should be 0 for certain prediction."""
        probs = [1.0, 0.0, 0.0, 0.0]
        entropy = DriftDetector.entropy(probs)
        assert entropy < 0.001
    
    def test_entropy_change_positive(self):
        """Entropy change should be positive when uncertainty increases."""
        baseline_mean = 0.5
        current_entropies = [0.8, 0.9, 0.7, 0.85]  # Higher than baseline
        
        change = DriftDetector.entropy_change(current_entropies, baseline_mean)
        assert change > 0  # Positive = more uncertainty
    
    def test_latency_drift_regression(self):
        """Latency drift should detect slowdown."""
        baseline_p50 = 50.0
        baseline_p99 = 100.0
        current = [80.0, 90.0, 100.0, 150.0, 200.0]  # Slower
        
        p50_change, p99_change = DriftDetector.latency_drift(
            current, baseline_p50, baseline_p99
        )
        assert p50_change > 0  # Positive = slower
        assert p99_change > 0
    
    def test_compute_prediction_distribution(self):
        """Should compute correct class distribution."""
        predictions = ["a", "a", "a", "b", "c"]
        dist = DriftDetector.compute_prediction_distribution(predictions)
        
        assert dist["a"] == 0.6
        assert dist["b"] == 0.2
        assert dist["c"] == 0.2
    
    def test_compute_centroid(self):
        """Should compute correct centroid."""
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        centroid = DriftDetector.compute_centroid(embeddings)
        
        assert len(centroid) == 2
        assert abs(centroid[0] - 2/3) < 0.001
        assert abs(centroid[1] - 2/3) < 0.001
