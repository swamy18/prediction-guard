"""
Tests for the middleware helpers.
"""

import pytest
import numpy as np

from prediction_guard.middleware.helpers import (
    compute_input_hash,
    compute_embedding_summary,
    compute_entropy,
    extract_confidence,
)


class TestComputeInputHash:
    """Tests for input hashing."""
    
    def test_string_hashing(self):
        """String input should be hashed."""
        hash1 = compute_input_hash("hello world")
        hash2 = compute_input_hash("hello world")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex
    
    def test_dict_hashing(self):
        """Dict input should be JSON-serialized and hashed."""
        hash1 = compute_input_hash({"key": "value"})
        hash2 = compute_input_hash({"key": "value"})
        assert hash1 == hash2
    
    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes."""
        hash1 = compute_input_hash("input1")
        hash2 = compute_input_hash("input2")
        assert hash1 != hash2
    
    def test_dict_key_order_invariant(self):
        """Dict hashing should be order-invariant."""
        hash1 = compute_input_hash({"a": 1, "b": 2})
        hash2 = compute_input_hash({"b": 2, "a": 1})
        assert hash1 == hash2


class TestComputeEmbeddingSummary:
    """Tests for embedding summarization."""
    
    def test_single_embedding_unchanged(self):
        """Single embedding should be returned as-is."""
        emb = [0.1, 0.2, 0.3]
        summary = compute_embedding_summary(emb)
        assert summary == emb
    
    def test_batch_embedding_centroid(self):
        """Batch of embeddings should return centroid."""
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        summary = compute_embedding_summary(embeddings)
        assert len(summary) == 2
        assert abs(summary[0] - 0.5) < 0.001
        assert abs(summary[1] - 0.5) < 0.001


class TestComputeEntropy:
    """Tests for entropy computation."""
    
    def test_uniform_distribution(self):
        """Uniform distribution should have max entropy."""
        probs = [0.25, 0.25, 0.25, 0.25]
        entropy = compute_entropy(probs)
        expected = np.log(4)
        assert abs(entropy - expected) < 0.001
    
    def test_certain_distribution(self):
        """Certain (one-hot) distribution should have near-zero entropy."""
        probs = [1.0, 0.0, 0.0]
        entropy = compute_entropy(probs)
        assert entropy < 0.001
    
    def test_binary_distribution(self):
        """Binary distribution should work correctly."""
        probs = [0.8, 0.2]
        entropy = compute_entropy(probs)
        assert entropy > 0


class TestExtractConfidence:
    """Tests for confidence extraction."""
    
    def test_direct_float(self):
        """Direct float should be returned."""
        assert extract_confidence(0.85) == 0.85
    
    def test_dict_with_confidence(self):
        """Dict with 'confidence' key should work."""
        output = {"confidence": 0.9, "prediction": "A"}
        assert extract_confidence(output) == 0.9
    
    def test_dict_with_score(self):
        """Dict with 'score' key should work."""
        output = {"score": 0.75}
        assert extract_confidence(output) == 0.75
    
    def test_list_of_probabilities(self):
        """List of probabilities should return max."""
        output = [0.1, 0.3, 0.6]
        assert extract_confidence(output) == 0.6
    
    def test_fallback_value(self):
        """Unknown format should return 0.5."""
        output = "unknown"
        assert extract_confidence(output) == 0.5
