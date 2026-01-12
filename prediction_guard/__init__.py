"""
Prediction Guard - ML Model Failure Detection and Rollback System.

A lightweight middleware that detects ML model failure and decides
when to roll back. This is a decision system, not a monitoring dashboard.

Quick Start:
    >>> from prediction_guard import PredictionGuard
    >>> guard = PredictionGuard()
    >>> decision = guard.analyze_and_decide()
    >>> print(decision.state)

For middleware integration:
    >>> from prediction_guard.middleware import PredictionInterceptor
    >>> from prediction_guard.types import GuardConfig
    >>> config = GuardConfig(current_model_version="v2.0")
    >>> interceptor = PredictionInterceptor(config)

Key Components:
    - PredictionGuard: Main orchestrator (analyze, decide, act)
    - PredictionInterceptor: FastAPI-compatible middleware
    - GuardConfig: Configuration with all thresholds
    - HealthDecision: Output with state, reasons, and action

Design Principles:
    - Decision-first: Every analysis leads to explicit decision
    - Multi-signal: Drift alone is not enough to trigger action
    - Privacy-safe: Never log raw user data
    - Human-in-the-loop: Auto-rollback is off by default

Author: Prediction Guard Team
Version: 0.1.0
License: MIT
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Prediction Guard Team"
__license__ = "MIT"

# Core types (public API)
# Configuration
from prediction_guard.config import load_config, save_config

# Main orchestrator
from prediction_guard.guard import PredictionGuard
from prediction_guard.types import (
    ActionType,
    AnalysisResult,
    DriftMetric,
    DriftType,
    GuardConfig,
    HealthDecision,
    IncidentSnapshot,
    ModelHealthState,
    PredictionEvent,
    RollbackAction,
    RollbackMechanism,
)

__all__ = [
    "ActionType",
    "AnalysisResult",
    "DriftMetric",
    "DriftType",
    # Configuration
    "GuardConfig",
    "HealthDecision",
    "IncidentSnapshot",
    # Types - Enums
    "ModelHealthState",
    # Types - Data classes
    "PredictionEvent",
    # Main class
    "PredictionGuard",
    "RollbackAction",
    "RollbackMechanism",
    "__author__",
    "__license__",
    # Version info
    "__version__",
    "load_config",
    "save_config",
]
