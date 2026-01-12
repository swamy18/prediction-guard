"""
Example: Using Prediction Guard with a mock ML model.

This example demonstrates:
1. Setting up the middleware
2. Logging predictions
3. Creating a baseline
4. Running analysis
5. Making health decisions
"""

import random
import time
from datetime import datetime, timedelta

# Add parent to path for running without install
import sys
sys.path.insert(0, "..")

from prediction_guard.types import GuardConfig
from prediction_guard.middleware import PredictionInterceptor
from prediction_guard.guard import PredictionGuard


def mock_model_predict(input_data: dict) -> dict:
    """Simulates an ML model prediction."""
    # Simulate some latency
    time.sleep(random.uniform(0.01, 0.05))
    
    # Generate mock prediction
    confidence = random.uniform(0.7, 0.99)
    prediction = "positive" if random.random() > 0.3 else "negative"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": [confidence, 1 - confidence],
        "embedding": [random.random() for _ in range(8)],
    }


def main():
    # Create configuration
    config = GuardConfig(
        log_directory="./demo_logs",
        baseline_directory="./demo_baselines",
        incident_directory="./demo_incidents",
        current_model_version="v1.0",
        fallback_model_version="v0.9",
        min_samples_for_analysis=10,  # Low for demo
    )
    
    # Create interceptor
    interceptor = PredictionInterceptor(config, buffer_size=1)
    
    print("=== Generating Predictions ===")
    
    # Generate some predictions
    for i in range(50):
        input_data = {"user_id": f"user_{i}", "query": f"sample query {i}"}
        
        with interceptor.intercept(input_data, {"region": "us-east"}) as ctx:
            result = mock_model_predict(input_data)
            ctx.set_result(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                embedding=result["embedding"],
            )
    
    interceptor.close()
    print(f"Logged 50 predictions to {config.log_directory}")
    
    # Create guard and baseline
    print("\n=== Creating Baseline ===")
    guard = PredictionGuard(config)
    
    # Manually create baseline from the logs we just generated
    from prediction_guard.logging import LogReader
    log_reader = LogReader(config.log_directory)
    events = log_reader.read_window(hours=1)
    
    if events:
        guard.baseline_manager.compute_baseline_from_events(events, "v1.0")
        print(f"Created baseline from {len(events)} events")
    
    # Run analysis
    print("\n=== Running Analysis ===")
    analysis = guard.analyze()
    
    if analysis:
        print(f"Sample count: {analysis.sample_count}")
        print(f"Feature drift: {analysis.feature_drift_score:.4f}")
        print(f"Embedding drift: {analysis.embedding_drift_score:.4f}")
        print(f"Prediction drift: {analysis.prediction_drift_score:.4f}")
        
        # Make decision
        print("\n=== Making Decision ===")
        decision = guard.decide(analysis)
        
        from prediction_guard.decision import DecisionEngine
        engine = DecisionEngine(config)
        print(engine.explain_decision(decision))
    else:
        print("Insufficient data for analysis")
    
    # Show status
    print("\n=== System Status ===")
    import json
    print(json.dumps(guard.get_status(), indent=2))


if __name__ == "__main__":
    main()
