"""
Log Reader - Efficient reading and filtering of JSONL logs.

Supports:
- Time-windowed reads
- Model version filtering
- Memory-efficient streaming
- Replay capability
"""

import json

from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path

from ..types import PredictionEvent


class LogReader:
    """
    Reads and filters prediction events from JSONL logs.

    Usage:
        reader = LogReader("./logs")
        events = reader.read_window(hours=1)
    """

    def __init__(self, log_directory: str):
        """
        Initialize the log reader.

        Args:
            log_directory: Directory containing log files
        """
        self.log_directory = Path(log_directory)

    def read_window(
        self,
        hours: int | None = None,
        minutes: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model_version: str | None = None,
    ) -> list[PredictionEvent]:
        """
        Read events within a time window.

        Args:
            hours: Window size in hours (alternative to start/end)
            minutes: Window size in minutes (alternative to start/end)
            start_time: Explicit start time
            end_time: Explicit end time (defaults to now)
            model_version: Filter by model version

        Returns:
            List of PredictionEvent objects
        """
        return list(self.stream_window(
            hours=hours,
            minutes=minutes,
            start_time=start_time,
            end_time=end_time,
            model_version=model_version,
        ))

    def stream_window(
        self,
        hours: int | None = None,
        minutes: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model_version: str | None = None,
    ) -> Generator[PredictionEvent, None, None]:
        """
        Stream events within a time window (memory efficient).

        Same arguments as read_window, but yields events one by one.
        """
        # Determine time bounds
        if end_time is None:
            end_time = datetime.now()

        if start_time is None:
            if hours is not None:
                start_time = end_time - timedelta(hours=hours)
            elif minutes is not None:
                start_time = end_time - timedelta(minutes=minutes)
            else:
                # Default to last hour
                start_time = end_time - timedelta(hours=1)

        # Get relevant log files
        log_files = self._get_files_for_window(start_time, end_time)

        # Stream events from each file
        for log_file in log_files:
            yield from self._stream_file(log_file, start_time, end_time, model_version)

    def _get_files_for_window(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Path]:
        """Get log files that might contain events in the window."""
        if not self.log_directory.exists():
            return []

        # Get all JSONL files
        all_files = sorted(self.log_directory.glob("predictions*.jsonl"))

        relevant_files = []
        for f in all_files:
            # Try to extract date from filename
            try:
                # Format: predictions_YYYY-MM-DD.jsonl
                date_str = f.stem.replace("predictions_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                # Include if file date overlaps with window
                file_end = file_date.replace(hour=23, minute=59, second=59)
                if file_date <= end_time and file_end >= start_time:
                    relevant_files.append(f)
            except ValueError:
                # Non-dated file, always include
                relevant_files.append(f)

        return relevant_files

    def _stream_file(
        self,
        log_file: Path,
        start_time: datetime,
        end_time: datetime,
        model_version: str | None,
    ) -> Generator[PredictionEvent, None, None]:
        """Stream events from a single log file."""
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    event = PredictionEvent.from_dict(data)

                    # Time filter
                    if event.timestamp < start_time or event.timestamp > end_time:
                        continue

                    # Model version filter
                    if model_version and event.model_version != model_version:
                        continue

                    yield event

                except (json.JSONDecodeError, KeyError):
                    # Log corruption - skip but note it
                    # In production, would log this to a separate error log
                    continue

    def count_events(
        self,
        hours: int | None = None,
        minutes: int | None = None,
        model_version: str | None = None,
    ) -> int:
        """Count events in a window without loading all into memory."""
        count = 0
        for _ in self.stream_window(hours=hours, minutes=minutes, model_version=model_version):
            count += 1
        return count

    def get_latest_event(self, model_version: str | None = None) -> PredictionEvent | None:
        """Get the most recent event."""
        latest = None
        for event in self.stream_window(hours=24, model_version=model_version):
            if latest is None or event.timestamp > latest.timestamp:
                latest = event
        return latest
