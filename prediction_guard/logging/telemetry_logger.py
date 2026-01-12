"""
Telemetry Logger - Append-only JSONL logging for prediction events.

Design principles:
- Never lose data (append-only, fsync on write)
- Never log raw user data (only hashes and summaries)
- Always replayable (structured JSONL)
- Minimal overhead (buffered writes)
"""

import json
import os
import threading

from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from ..types import PredictionEvent


class TelemetryLogger:
    """
    Thread-safe, append-only JSONL logger for prediction telemetry.

    Usage:
        logger = TelemetryLogger("./logs")
        logger.log(event)
    """

    def __init__(
        self,
        log_directory: str,
        buffer_size: int = 10,
        rotate_daily: bool = True,
    ):
        """
        Initialize the telemetry logger.

        Args:
            log_directory: Directory for log files
            buffer_size: Number of events to buffer before flush
            rotate_daily: If True, creates new file each day
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.rotate_daily = rotate_daily

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._current_date: str | None = None
        self._file_handle: TextIO | None = None

    def log(self, event: PredictionEvent) -> None:
        """
        Log a prediction event.

        Thread-safe. Buffers writes for efficiency.
        """
        with self._lock:
            self._buffer.append(event.to_dict())

            if len(self._buffer) >= self.buffer_size:
                self._flush()

    def flush(self) -> None:
        """Force flush the buffer to disk."""
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        """Internal flush - must hold lock."""
        if not self._buffer:
            return

        # Get or create file handle
        handle = self._get_file_handle()

        # Write all buffered events
        for event_dict in self._buffer:
            line = json.dumps(event_dict, separators=(",", ":"))
            handle.write(line + "\n")

        # Ensure durability
        handle.flush()
        os.fsync(handle.fileno())

        self._buffer.clear()

    def _get_file_handle(self) -> TextIO:
        """Get current file handle, rotating if necessary."""
        today = datetime.now().strftime("%Y-%m-%d")

        if self.rotate_daily and self._current_date != today:
            if self._file_handle:
                self._file_handle.close()
            self._current_date = today
            self._file_handle = None

        if self._file_handle is None:
            if self.rotate_daily:
                filename = f"predictions_{today}.jsonl"
            else:
                filename = "predictions.jsonl"

            filepath = self.log_directory / filename
            self._file_handle = open(filepath, "a", encoding="utf-8")

        return self._file_handle

    def close(self) -> None:
        """Flush and close the logger."""
        with self._lock:
            self._flush()
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    @property
    def current_log_path(self) -> Path | None:
        """Get the current log file path."""
        if self.rotate_daily:
            today = datetime.now().strftime("%Y-%m-%d")
            return self.log_directory / f"predictions_{today}.jsonl"
        return self.log_directory / "predictions.jsonl"
