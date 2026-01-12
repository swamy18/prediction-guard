"""
Helper functions for middleware.

Utilities for hashing, embedding summarization, and entropy computation.
Never stores raw data - only privacy-safe summaries.
"""

import hashlib
import json

from typing import Any, cast

import numpy as np


def compute_input_hash(input_data: Any) -> str:
    """
    Compute SHA256 hash of input data.

    NEVER stores raw input - only the hash.

    Args:
        input_data: Any JSON-serializable input

    Returns:
        Hex-encoded SHA256 hash
    """
    # Normalize to string for consistent hashing
    if isinstance(input_data, str):
        data_str = input_data
    else:
        # Sort keys for consistent ordering
        data_str = json.dumps(input_data, sort_keys=True, default=str)

    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def compute_embedding_summary(embedding: list[float] | np.ndarray[Any, Any]) -> list[float]:
    """
    Summarize an embedding to a smaller representation.

    For a single embedding, returns as-is (already a summary).
    For multiple embeddings, returns the mean (centroid).

    Args:
        embedding: Single embedding or batch of embeddings

    Returns:
        Summary embedding (list of floats)
    """
    arr = np.array(embedding)

    # If 1D, it's already a summary
    if arr.ndim == 1:
        return cast("list[float]", arr.tolist())

    # If 2D, compute centroid
    if arr.ndim == 2:
        return cast("list[float]", np.mean(arr, axis=0).tolist())

    # Fallback: flatten and take mean of chunks
    flat = arr.flatten()
    return [float(np.mean(flat))]


def compute_entropy(probabilities: list[float] | np.ndarray[Any, Any]) -> float:
    """
    Compute Shannon entropy from probability distribution.

    Higher entropy = more uncertainty in prediction.

    Args:
        probabilities: Probability distribution (should sum to 1)

    Returns:
        Entropy value (non-negative)
    """
    probs = np.array(probabilities)

    # Handle edge cases
    if probs.size == 0:
        return 0.0

    # Normalize if needed
    prob_sum = np.sum(probs)
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        return 0.0

    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    return float(-np.sum(probs * np.log(probs)))


def extract_confidence(prediction_output: Any) -> float:
    """
    Extract confidence score from various prediction formats.

    Handles common formats:
    - Direct float
    - Dict with 'confidence', 'score', 'probability' key
    - List of probabilities (returns max)

    Args:
        prediction_output: Raw prediction output

    Returns:
        Confidence score [0, 1]
    """
    if isinstance(prediction_output, (int, float)):
        return float(min(1.0, max(0.0, prediction_output)))

    if isinstance(prediction_output, dict):
        for key in ["confidence", "score", "probability", "prob"]:
            if key in prediction_output:
                return float(prediction_output[key])
        # Check for nested structure
        if "probabilities" in prediction_output:
            probs = prediction_output["probabilities"]
            if isinstance(probs, list):
                return float(max(probs))

    if isinstance(prediction_output, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in prediction_output):
            return float(max(prediction_output))

    # Default fallback
    return 0.5
