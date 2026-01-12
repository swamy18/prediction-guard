"""
Example: Simulating model drift and watching Prediction Guard react.

This demonstrates:
1. Normal operation (healthy state)
2. Gradual drift (suspicious state)
3. Severe drift (unstable state, rollback recommended)
"""

import random
import time
import json
import sys
sys.path.insert(0, "..")

from prediction_guard.types import GuardConfig
from prediction_guard.middleware import PredictionInterceptor
from prediction_guard.guard import PredictionGuard
from prediction_guard.decision import DecisionEngine


def generate_normal_prediction():
    """Normal, healthy predictions."""
    return {
        "prediction": "positive" if random.random() > 0.3 else "negative",
        "confidence": random.uniform(0.85, 0.98),
        "embedding": [random.gauss(0.5, 0.1) for _ in range(8)],
    }


def generate_drifted_prediction():
    """Drifted predictions - shifted distribution."""
    return {
        "prediction": "negative" if random.random() > 0.2 else "positive",  # Flipped!
        "confidence": random.uniform(0.4, 0.7),  # Lower confidence
        "embedding": [random.gauss(0.8, 0.2) for _ in range(8)],  # Shifted centroid
    }


def log_predictions(interceptor, generator, count, label):
    """Generate and log predictions."""
    print(f"\nGenerating {count} {label} predictions...")
    for i in range(count):
        result = generator()
        interceptor.log_prediction(
            input_data=f"input_{i}",
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=[result["confidence"], 1 - result["confidence"]],
            embedding=result["embedding"],
            request_context={"type": label},
            latency_ms=random.uniform(10, 50),
        )
    interceptor.flush()


def main():
    config = GuardConfig(
        log_directory="./drift_demo_logs",
        baseline_directory="./drift_demo_baselines",
        incident_directory="./drift_demo_incidents",
        current_model_version="v1.0",
        min_samples_for_analysis=20,
        feature_drift_threshold=0.1,  # Sensitive thresholds for demo
        embedding_drift_threshold=0.1,
        prediction_drift_threshold=0.05,
    )
    
    interceptor = PredictionInterceptor(config, buffer_size=1)
    guard = PredictionGuard(config)
    engine = DecisionEngine(config)
    
    # Phase 1: Generate normal predictions and create baseline
    print("=" * 60)
    print("PHASE 1: NORMAL OPERATION - Creating Baseline")
    print("=" * 60)
    
    log_predictions(interceptor, generate_normal_prediction, 100, "normal")
    
    from prediction_guard.logging import LogReader
    log_reader = LogReader(config.log_directory)
    events = log_reader.read_window(hours=1)
    guard.baseline_manager.compute_baseline_from_events(events, "v1.0")
    print(f"Baseline created from {len(events)} events")
    
    # Analyze - should be healthy
    decision = guard.analyze_and_decide()
    if decision:
        print(f"\nDecision: {decision.state.value.upper()}")
        print(f"Reasons: {decision.reasons}")
    
    # Phase 2: Add some drift
    print("\n" + "=" * 60)
    print("PHASE 2: INTRODUCING DRIFT")
    print("=" * 60)
    
    # Mix normal and drifted predictions
    for i in range(50):
        if random.random() > 0.5:
            result = generate_drifted_prediction()
        else:
            result = generate_normal_prediction()
        
        interceptor.log_prediction(
            input_data=f"mixed_input_{i}",
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=[result["confidence"], 1 - result["confidence"]],
            embedding=result["embedding"],
            request_context={"type": "mixed"},
            latency_ms=random.uniform(10, 100),
        )
    interceptor.flush()
    
    decision = guard.analyze_and_decide()
    if decision:
        print(f"\nDecision: {decision.state.value.upper()}")
        print(f"Reasons: {decision.reasons}")
        print(f"Recommended Action: {decision.recommended_action.value}")
    
    # Phase 3: Severe drift
    print("\n" + "=" * 60)
    print("PHASE 3: SEVERE DRIFT - MODEL DEGRADATION")
    print("=" * 60)
    
    log_predictions(interceptor, generate_drifted_prediction, 100, "drifted")
    
    decision = guard.analyze_and_decide()
    if decision:
        print("\n" + engine.explain_decision(decision))
        
        # Run full pipeline
        result = guard.run_pipeline()
        print("\nPipeline Result:")
        print(json.dumps(result, indent=2))
    
    interceptor.close()


if __name__ == "__main__":
    main()
