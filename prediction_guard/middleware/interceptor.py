"""
Prediction Interceptor - FastAPI-compatible middleware for telemetry.

This module provides the middleware layer that wraps ML inference
endpoints to log structured telemetry with minimal latency impact.

Key Features:
- Context manager API for clean integration
- Direct logging API for flexibility
- Thread-safe buffered writes
- Privacy-preserving (never stores raw input)

Latency Target: <5ms overhead

Usage with FastAPI:
    >>> from prediction_guard.middleware import PredictionInterceptor
    >>> from prediction_guard.types import GuardConfig
    >>>
    >>> config = GuardConfig(current_model_version="v2.0")
    >>> interceptor = PredictionInterceptor(config)
    >>>
    >>> @app.post("/predict")
    >>> async def predict(request: PredictRequest):
    ...     with interceptor.intercept(request.data) as ctx:
    ...         result = model.predict(request.data)
    ...         ctx.set_result(result.prediction, result.confidence)
    ...     return result

Author: Prediction Guard Team
"""

from __future__ import annotations

import logging
import time
import uuid

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..logging import TelemetryLogger
from ..types import GuardConfig, PredictionEvent
from .helpers import compute_embedding_summary, compute_entropy, compute_input_hash

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_BUFFER_SIZE = 10
LATENCY_WARNING_THRESHOLD_MS = 10.0


# =============================================================================
# INTERCEPT CONTEXT
# =============================================================================

@dataclass
class InterceptContext:
    """
    Context for a single prediction interception.

    This context object is yielded by the intercept() context manager.
    The caller sets the prediction result, and the context manager
    handles logging automatically.

    Attributes:
        request_id: Unique identifier for this request
        start_time: High-precision start time for latency measurement
        input_hash: SHA256 hash of the input (never raw data)
        request_context: Optional metadata (region, user_type, etc.)
        prediction: Set by caller - the prediction result
        confidence_score: Set by caller - primary confidence metric
        prediction_entropy: Set by caller or computed - uncertainty measure
        embedding_summary: Set by caller - mean embedding vector
    """

    request_id: str
    start_time: float
    input_hash: str
    request_context: dict[str, Any]

    # Set by caller via set_result()
    prediction: Any = None
    confidence_score: float = 0.0
    prediction_entropy: float = 0.0
    embedding_summary: tuple[float, ...] = field(default_factory=tuple)

    # Internal flag
    _result_set: bool = field(default=False, repr=False)

    def set_result(
        self,
        prediction: Any,
        confidence: float,
        probabilities: list[float] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """
        Set the prediction result for logging.

        Args:
            prediction: The model's prediction output
            confidence: Primary confidence score [0, 1]
            probabilities: Optional probability distribution for entropy calculation
            embedding: Optional embedding vector for drift detection

        Raises:
            ValueError: If called more than once
        """
        if self._result_set:
            raise ValueError(
                "set_result() called multiple times. "
                "Only call once per prediction."
            )

        self.prediction = prediction
        self.confidence_score = min(1.0, max(0.0, confidence))

        # Compute entropy from probabilities or confidence
        if probabilities:
            self.prediction_entropy = compute_entropy(probabilities)
        elif 0 < confidence < 1:
            # Binary entropy from confidence
            self.prediction_entropy = compute_entropy([confidence, 1 - confidence])
        else:
            self.prediction_entropy = 0.0

        # Summarize embedding if provided
        if embedding:
            self.embedding_summary = tuple(compute_embedding_summary(embedding))

        self._result_set = True

    @property
    def has_result(self) -> bool:
        """True if set_result() has been called."""
        return self._result_set


# =============================================================================
# PREDICTION INTERCEPTOR
# =============================================================================

class PredictionInterceptor:
    """
    Intercepts predictions to log telemetry with minimal overhead.

    This is the primary integration point for inference services.
    It provides two APIs:

    1. Context Manager (recommended):
        >>> with interceptor.intercept(input_data) as ctx:
        ...     result = model.predict(input_data)
        ...     ctx.set_result(result.prediction, result.confidence)

    2. Direct Logging:
        >>> request_id = interceptor.log_prediction(
        ...     input_data=data,
        ...     prediction=result,
        ...     confidence=0.92,
        ... )

    Thread Safety:
        The interceptor is thread-safe. Multiple threads can call
        intercept() and log_prediction() concurrently.

    Latency:
        Target overhead is <5ms. If telemetry logging takes longer,
        a warning is logged.
    """

    __slots__ = (
        "_buffer_size",
        "_config",
        "_logger",
        "_model_version",
    )

    def __init__(
        self,
        config: GuardConfig,
        model_version: str | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        """
        Initialize the interceptor.

        Args:
            config: Guard configuration with log directory and version
            model_version: Override model version (defaults to config value)
            buffer_size: Events to buffer before flushing (default: 10)
        """
        self._config = config
        self._model_version = model_version or config.current_model_version
        self._buffer_size = buffer_size

        self._logger = TelemetryLogger(
            config.log_directory,
            buffer_size=buffer_size,
        )

        logger.debug(
            f"PredictionInterceptor initialized: "
            f"version={self._model_version}, buffer_size={buffer_size}"
        )

    @property
    def model_version(self) -> str:
        """Current model version being logged."""
        return self._model_version

    @contextmanager
    def intercept(
        self,
        input_data: Any,
        request_context: dict[str, Any] | None = None,
    ) -> Generator[InterceptContext, None, None]:
        """
        Context manager for intercepting a prediction.

        Automatically measures latency and logs the event when
        the context exits.

        Args:
            input_data: Raw input data (will be hashed, not stored)
            request_context: Optional metadata (region, user_type, etc.)

        Yields:
            InterceptContext for setting prediction results

        Example:
            >>> with interceptor.intercept(input_data) as ctx:
            ...     result = model.predict(input_data)
            ...     ctx.set_result(result.prediction, result.confidence)
        """
        ctx = InterceptContext(
            request_id=str(uuid.uuid4()),
            start_time=time.perf_counter(),
            input_hash=compute_input_hash(input_data),
            request_context=request_context or {},
        )

        try:
            yield ctx
        finally:
            # Compute latency
            end_time = time.perf_counter()
            latency_ms = (end_time - ctx.start_time) * 1000

            # Warn if overhead is too high
            if latency_ms > LATENCY_WARNING_THRESHOLD_MS:
                logger.warning(
                    f"Prediction latency ({latency_ms:.1f}ms) exceeds target "
                    f"({LATENCY_WARNING_THRESHOLD_MS}ms)"
                )

            # Log event
            self._log_event(ctx, latency_ms)

    def _log_event(self, ctx: InterceptContext, latency_ms: float) -> None:
        """
        Create and log a prediction event from context.

        Args:
            ctx: Intercept context with result
            latency_ms: Measured latency
        """
        if not ctx.has_result:
            logger.warning(
                f"Request {ctx.request_id}: set_result() was not called. "
                "Logging with default values."
            )

        event = PredictionEvent(
            timestamp=datetime.now(),
            model_version=self._model_version,
            request_id=ctx.request_id,
            input_hash=ctx.input_hash,
            embedding_summary=ctx.embedding_summary,
            prediction=ctx.prediction,
            confidence_score=ctx.confidence_score,
            prediction_entropy=ctx.prediction_entropy,
            latency_ms=latency_ms,
            request_context=ctx.request_context,
        )

        self._logger.log(event)

    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        confidence: float,
        probabilities: list[float] | None = None,
        embedding: list[float] | None = None,
        request_context: dict[str, Any] | None = None,
        latency_ms: float | None = None,
    ) -> str:
        """
        Directly log a prediction event.

        Alternative to the context manager for cases where the
        context manager pattern doesn't fit.

        Args:
            input_data: Raw input data (will be hashed, not stored)
            prediction: Model's prediction output
            confidence: Primary confidence score [0, 1]
            probabilities: Optional probability distribution
            embedding: Optional embedding vector
            request_context: Optional metadata
            latency_ms: Measured latency (if already known)

        Returns:
            request_id for this prediction

        Example:
            >>> request_id = interceptor.log_prediction(
            ...     input_data="test input",
            ...     prediction="positive",
            ...     confidence=0.92,
            ... )
        """
        request_id = str(uuid.uuid4())

        # Compute entropy
        if probabilities:
            entropy = compute_entropy(probabilities)
        elif 0 < confidence < 1:
            entropy = compute_entropy([confidence, 1 - confidence])
        else:
            entropy = 0.0

        # Summarize embedding
        embedding_summary: tuple[float, ...]
        if embedding:
            embedding_summary = tuple(compute_embedding_summary(embedding))
        else:
            embedding_summary = ()

        event = PredictionEvent(
            timestamp=datetime.now(),
            model_version=self._model_version,
            request_id=request_id,
            input_hash=compute_input_hash(input_data),
            embedding_summary=embedding_summary,
            prediction=prediction,
            confidence_score=min(1.0, max(0.0, confidence)),
            prediction_entropy=entropy,
            latency_ms=latency_ms or 0.0,
            request_context=request_context or {},
        )

        self._logger.log(event)
        logger.debug(f"Logged prediction: {request_id}")

        return request_id

    def flush(self) -> None:
        """
        Flush pending logs to disk.

        Call this periodically in long-running processes or
        before shutdown to ensure all events are persisted.
        """
        self._logger.flush()
        logger.debug("Interceptor buffer flushed")

    def close(self) -> None:
        """
        Close the interceptor and flush all pending logs.

        Call this during application shutdown.
        """
        self._logger.close()
        logger.info("Interceptor closed")

    def __enter__(self) -> PredictionInterceptor:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PredictionInterceptor("
            f"model_version={self._model_version!r}, "
            f"buffer_size={self._buffer_size})"
        )
