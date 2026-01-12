"""
Action Executor - Executes rollback actions with safeguards.

Supports multiple rollback mechanisms:
- Model alias switch
- Environment variable change
- Config file update
- Feature flag toggle

All actions are logged, reversible, and respect cooldowns.
"""

import json
import os
import uuid

from datetime import datetime
from pathlib import Path

from ..types import (
    ActionType,
    GuardConfig,
    HealthDecision,
    RollbackAction,
    RollbackMechanism,
)
from .cooldown import CooldownManager


class ActionExecutor:
    """
    Executes and logs rollback actions with full safeguards.
    """

    def __init__(self, config: GuardConfig):
        self.config = config
        self.cooldown = CooldownManager(config.rollback_cooldown_minutes)
        self.action_log_path = Path(config.incident_directory) / "actions.jsonl"
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rollback handlers by mechanism
        self._handlers = {
            RollbackMechanism.CONFIG_FILE: self._rollback_config_file,
            RollbackMechanism.ENV_VAR: self._rollback_env_var,
            RollbackMechanism.MODEL_ALIAS: self._rollback_model_alias,
            RollbackMechanism.FEATURE_FLAG: self._rollback_feature_flag,
        }

    def execute(
        self,
        decision: HealthDecision,
        force: bool = False,
    ) -> RollbackAction | None:
        """
        Execute a rollback if warranted by the decision.

        Args:
            decision: Health decision from DecisionEngine
            force: If True, bypass auto-rollback check and cooldown

        Returns:
            RollbackAction if executed, None if skipped
        """
        # Check if action is warranted
        if decision.recommended_action != ActionType.ROLLBACK:
            return None

        # Check auto-rollback setting
        if not force and not self.config.auto_rollback_enabled:
            self._log_skipped_action(decision, "auto_rollback_disabled")
            return None

        # Check cooldown
        if not force and self.cooldown.is_cooling_down():
            self._log_skipped_action(decision, "cooldown_active")
            return None

        # Execute rollback
        action = RollbackAction(
            action_id=str(uuid.uuid4()),
            from_version=self.config.current_model_version,
            to_version=self.config.fallback_model_version,
            mechanism=self.config.rollback_mechanism,
            reason="; ".join(decision.reasons),
            initiated_at=datetime.now(),
            is_auto=not force,
        )

        try:
            handler = self._handlers[self.config.rollback_mechanism]
            handler(action)
            action.success = True
            action.completed_at = datetime.now()
            self.cooldown.start_cooldown()
        except Exception as e:
            action.success = False
            action.error_message = str(e)
            action.completed_at = datetime.now()

        self._log_action(action)
        return action

    def _rollback_config_file(self, action: RollbackAction) -> None:
        """Update config file to switch model version."""
        config_path = os.environ.get("PREDICTION_GUARD_CONFIG", "./prediction_guard_config.json")

        if Path(config_path).exists():
            with open(config_path) as f:
                data = json.load(f)
        else:
            data = {}

        data["current_model_version"] = action.to_version
        data["_rollback_at"] = action.initiated_at.isoformat()
        data["_rollback_from"] = action.from_version

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def _rollback_env_var(self, action: RollbackAction) -> None:
        """Update environment variable for model version."""
        os.environ["MODEL_VERSION"] = action.to_version
        os.environ["MODEL_ROLLBACK_AT"] = action.initiated_at.isoformat()

    def _rollback_model_alias(self, action: RollbackAction) -> None:
        """Update model alias (placeholder for actual implementation)."""
        alias_path = Path(self.config.baseline_directory) / "model_alias.json"
        alias_path.parent.mkdir(parents=True, exist_ok=True)

        with open(alias_path, "w") as f:
            json.dump({
                "current_alias": action.to_version,
                "previous_alias": action.from_version,
                "switched_at": action.initiated_at.isoformat(),
            }, f, indent=2)

    def _rollback_feature_flag(self, action: RollbackAction) -> None:
        """Update feature flag (placeholder for actual implementation)."""
        flag_path = Path(self.config.baseline_directory) / "feature_flags.json"
        flag_path.parent.mkdir(parents=True, exist_ok=True)

        if flag_path.exists():
            with open(flag_path) as f:
                flags = json.load(f)
        else:
            flags = {}

        flags["active_model_version"] = action.to_version
        flags["model_rollback_active"] = True
        flags["rollback_at"] = action.initiated_at.isoformat()

        with open(flag_path, "w") as f:
            json.dump(flags, f, indent=2)

    def _log_action(self, action: RollbackAction) -> None:
        """Log action to JSONL file."""
        with open(self.action_log_path, "a") as f:
            f.write(json.dumps(action.to_dict()) + "\n")

    def _log_skipped_action(self, decision: HealthDecision, reason: str) -> None:
        """Log when an action was skipped."""
        entry = {
            "type": "skipped_action",
            "timestamp": datetime.now().isoformat(),
            "model_version": decision.model_version,
            "skip_reason": reason,
            "decision_state": decision.state.value,
        }
        with open(self.action_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def revert_rollback(self, action_id: str) -> RollbackAction | None:
        """Revert a previous rollback (manual operation)."""
        # Swap current and fallback in the revert
        action = RollbackAction(
            action_id=str(uuid.uuid4()),
            from_version=self.config.fallback_model_version,
            to_version=self.config.current_model_version,
            mechanism=self.config.rollback_mechanism,
            reason=f"revert_of_{action_id}",
            initiated_at=datetime.now(),
            is_auto=False,
        )

        try:
            handler = self._handlers[self.config.rollback_mechanism]
            handler(action)
            action.success = True
            action.completed_at = datetime.now()
        except Exception as e:
            action.success = False
            action.error_message = str(e)
            action.completed_at = datetime.now()

        self._log_action(action)
        return action
