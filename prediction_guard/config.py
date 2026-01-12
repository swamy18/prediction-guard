"""
Configuration management for Prediction Guard.

Loads config from file with sensible defaults. No magic, no surprises.
"""

import json
import os

from pathlib import Path
from typing import Any

from .types import GuardConfig, RollbackMechanism

DEFAULT_CONFIG_PATH = "./prediction_guard_config.json"


def load_config(config_path: str | None = None) -> GuardConfig:
    """
    Load configuration from file, with fallback to defaults.

    Priority:
    1. Explicit config_path argument
    2. PREDICTION_GUARD_CONFIG environment variable
    3. Default path (./prediction_guard_config.json)
    4. Built-in defaults
    """
    path = config_path or os.environ.get("PREDICTION_GUARD_CONFIG", DEFAULT_CONFIG_PATH)

    if Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        return _dict_to_config(data)

    # Return defaults if no config file
    return GuardConfig()


def save_config(config: GuardConfig, config_path: str | None = None) -> None:
    """
    Save configuration to file.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(_config_to_dict(config), f, indent=2)


def _dict_to_config(data: dict[str, Any]) -> GuardConfig:
    """Convert dictionary to GuardConfig, handling enums."""
    if "rollback_mechanism" in data:
        data["rollback_mechanism"] = RollbackMechanism(data["rollback_mechanism"])
    return GuardConfig(**data)


def _config_to_dict(config: GuardConfig) -> dict[str, Any]:
    """Convert GuardConfig to dictionary for serialization."""
    return {
        "feature_drift_threshold": config.feature_drift_threshold,
        "embedding_drift_threshold": config.embedding_drift_threshold,
        "prediction_drift_threshold": config.prediction_drift_threshold,
        "confidence_entropy_threshold": config.confidence_entropy_threshold,
        "latency_p99_threshold_ms": config.latency_p99_threshold_ms,
        "analysis_window_minutes": config.analysis_window_minutes,
        "baseline_window_days": config.baseline_window_days,
        "min_samples_for_analysis": config.min_samples_for_analysis,
        "auto_rollback_enabled": config.auto_rollback_enabled,
        "rollback_cooldown_minutes": config.rollback_cooldown_minutes,
        "rollback_mechanism": config.rollback_mechanism.value,
        "log_directory": config.log_directory,
        "baseline_directory": config.baseline_directory,
        "incident_directory": config.incident_directory,
        "current_model_version": config.current_model_version,
        "fallback_model_version": config.fallback_model_version,
        "business_proxy_enabled": config.business_proxy_enabled,
        "business_proxy_threshold": config.business_proxy_threshold,
        "business_proxy_overrides_drift": config.business_proxy_overrides_drift,
    }


def create_default_config_file(path: str = DEFAULT_CONFIG_PATH) -> None:
    """Create a default configuration file for users to customize."""
    save_config(GuardConfig(), path)
    print(f"Created default config at: {path}")
