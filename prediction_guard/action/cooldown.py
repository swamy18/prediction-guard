"""
Cooldown Manager - Prevents rollback storms.

Enforces a minimum time between rollback actions.
"""

from datetime import datetime, timedelta


class CooldownManager:
    """
    Manages cooldown periods between rollback actions.
    """

    def __init__(self, cooldown_minutes: int):
        self.cooldown_minutes = cooldown_minutes
        self._last_action_at: datetime | None = None

    def start_cooldown(self) -> None:
        """Mark the start of a cooldown period."""
        self._last_action_at = datetime.now()

    def is_cooling_down(self) -> bool:
        """Check if we're currently in cooldown."""
        if self._last_action_at is None:
            return False

        cooldown_end = self._last_action_at + timedelta(minutes=self.cooldown_minutes)
        return datetime.now() < cooldown_end

    def remaining_seconds(self) -> float:
        """Get remaining cooldown time in seconds."""
        if not self.is_cooling_down():
            return 0.0

        # Type narrowing: is_cooling_down() guarantees _last_action_at is not None
        assert self._last_action_at is not None
        cooldown_end = self._last_action_at + timedelta(minutes=self.cooldown_minutes)
        remaining = cooldown_end - datetime.now()
        return max(0.0, remaining.total_seconds())

    def reset(self) -> None:
        """Reset the cooldown (manual override)."""
        self._last_action_at = None
