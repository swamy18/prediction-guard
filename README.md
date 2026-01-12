# Prediction Guard

<div align="center">

**A lightweight middleware for ML model failure detection and rollback.**

*This is a decision system, not a monitoring dashboard.*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-34%20passed-green.svg)]()

</div>

---

## 🎯 What is Prediction Guard?

Prediction Guard is a **thin middleware layer** that sits in front of your ML inference endpoint and:

1. **Logs** statistically useful prediction telemetry
2. **Analyzes** logs for drift and failure signals  
3. **Decides** on model health with explicit reasoning
4. **Acts** on decisions (rollback) with safeguards

**Key Insight**: Monitoring tells you something is wrong. Prediction Guard tells you **what to do about it**.

---

## 🧠 Core Philosophy

| Principle | What it means |
|-----------|---------------|
| **Decision-first** | Every analysis leads to an explicit decision with reasons |
| **Multi-signal required** | Drift alone is NOT enough to trigger rollback |
| **Privacy-safe** | Never log raw user data—only hashes and summaries |
| **Human-in-the-loop** | Auto-rollback is off by default; thresholds are manually tunable |
| **Explainable** | Every decision includes reasons a non-ML engineer can understand |
| **Minimal** | Only 2 dependencies: `numpy` and `scipy` |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/swamy18/prediction-guard.git
cd prediction-guard

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
```

That's it. No Kafka. No Redis. No heavy infrastructure.

---

## 🚀 Quick Start

### 1. Initialize Configuration

```bash
prediction-guard init
```

This creates `prediction_guard_config.json` with sensible defaults.

### 2. Integrate with Your Inference Endpoint

```python
from prediction_guard.middleware import PredictionInterceptor
from prediction_guard.types import GuardConfig

# Configure
config = GuardConfig(
    current_model_version="v2.0",
    fallback_model_version="v1.9",
    log_directory="./logs",
)

# Create interceptor
interceptor = PredictionInterceptor(config)

# In your prediction endpoint
def predict(input_data):
    with interceptor.intercept(input_data, {"region": "us-east"}) as ctx:
        result = your_model.predict(input_data)
        ctx.set_result(
            prediction=result.prediction,
            confidence=result.confidence,
            probabilities=result.probabilities,
            embedding=result.embedding,
        )
    return result
```

### 3. Create a Baseline (from historical data)

```bash
prediction-guard baseline create --model v2.0 --days 7
```

### 4. Run Analysis and Get Decision

```bash
prediction-guard decide --model v2.0
```

Output:
```
=== Model Health Decision ===
Model Version: v2.0
State: HEALTHY
Confidence: 95%
Recommended Action: none

Reasons:
  - No issues detected
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           YOUR APPLICATION                               │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐         │
│  │   Request   │ ──▶│  Interceptor     │ ──▶│   ML Model      │         │
│  │             │    │  (logs telemetry)│    │   Prediction    │         │
│  └─────────────┘    └────────┬─────────┘    └─────────────────┘         │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               │ Append-only writes
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        JSONL LOG FILES                                   │
│                                                                          │
│  predictions_2024-01-15.jsonl                                           │
│  predictions_2024-01-16.jsonl                                           │
│  ...                                                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               │ Scheduled / Manual trigger
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OFFLINE ANALYZER                                    │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Drift Detectors                                                   │  │
│  │  • Feature Drift (Kolmogorov-Smirnov test)                        │  │
│  │  • Embedding Drift (Cosine distance from baseline centroid)       │  │
│  │  • Prediction Drift (Population Stability Index)                  │  │
│  │  • Confidence Entropy (Shannon entropy change)                    │  │
│  │  • Latency Drift (P50/P99 percentile changes)                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Compares current window against stored baseline                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               │ AnalysisResult
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DECISION ENGINE                                     │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Multi-Signal Logic                                                │  │
│  │                                                                    │  │
│  │  if drift_signals >= 3:                                           │  │
│  │      state = UNSTABLE, action = ROLLBACK                          │  │
│  │                                                                    │  │
│  │  if drift_signals == 2 AND (embedding + confidence):              │  │
│  │      state = UNSTABLE, action = ROLLBACK                          │  │
│  │                                                                    │  │
│  │  if drift_signals == 1:                                           │  │
│  │      state = SUSPICIOUS, action = ALERT                           │  │
│  │                                                                    │  │
│  │  if business_proxy_healthy:                                       │  │
│  │      OVERRIDE drift signals → HEALTHY                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Output: HealthDecision with state, reasons, recommended_action         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               │ If action = ROLLBACK
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ACTION EXECUTOR                                     │
│                                                                          │
│  Rollback Mechanisms:                                                   │
│  • Config file update (prediction_guard_config.json)                    │
│  • Environment variable (MODEL_VERSION)                                 │
│  • Model alias file (model_alias.json)                                  │
│  • Feature flag file (feature_flags.json)                               │
│                                                                          │
│  Safeguards:                                                            │
│  ✓ Auto-rollback OFF by default                                        │
│  ✓ Cooldown period (30 min default)                                    │
│  ✓ All actions logged for audit                                        │
│  ✓ Revert capability                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Telemetry Events

Every prediction logs one structured event:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "model_version": "v2.0",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "input_hash": "a3f2b8c9d4e5f6...",
  "embedding_summary": [0.12, 0.34, 0.56, ...],
  "prediction": "positive",
  "confidence_score": 0.92,
  "prediction_entropy": 0.28,
  "latency_ms": 45.2,
  "request_context": {
    "region": "us-east-1",
    "user_type": "premium"
  }
}
```

### Privacy Guarantees

| Field | Privacy Treatment |
|-------|-------------------|
| `input_hash` | SHA256 hash of input—raw data NEVER stored |
| `embedding_summary` | Mean/centroid only—no individual embeddings |
| `request_context` | Optional metadata—you control what's included |

---

## 📈 Drift Detection Methods

### 1. Feature Drift (Kolmogorov-Smirnov Test)

Compares the distribution of a feature (e.g., confidence scores) between current window and baseline.

```python
from prediction_guard.analysis import DriftDetector

# Returns (ks_statistic, p_value)
stat, pvalue = DriftDetector.ks_test(current_values, baseline_values)

# Interpretation:
# stat > 0.15 AND pvalue < 0.05 → Significant drift
```

**When it fires**: Input data distribution has shifted (e.g., new user demographics)

### 2. Embedding Drift (Cosine Distance)

Measures how far the current embedding centroid has moved from baseline.

```python
distance = DriftDetector.cosine_distance(current_centroid, baseline_centroid)

# Interpretation:
# 0.0 = identical direction
# 1.0 = orthogonal
# 2.0 = opposite direction
```

**When it fires**: The semantic content of inputs has changed (e.g., new topics)

### 3. Prediction Drift (Population Stability Index)

Measures shift in prediction class distribution.

```python
psi = DriftDetector.population_stability_index(current_dist, baseline_dist)

# Interpretation:
# PSI < 0.1  → No significant change
# 0.1-0.25  → Moderate change, investigate
# PSI > 0.25 → Significant change, action needed
```

**When it fires**: Model is producing different class ratios than expected

### 4. Confidence Entropy

Measures change in prediction uncertainty.

```python
change = DriftDetector.entropy_change(current_entropies, baseline_mean)

# Interpretation:
# Positive = more uncertainty (model less confident)
# Negative = less uncertainty (could be overconfident)
```

**When it fires**: Model is becoming more/less certain about predictions

### 5. Latency Drift

Detects performance regression.

```python
p50_change, p99_change = DriftDetector.latency_drift(
    current_latencies, baseline_p50, baseline_p99
)

# Interpretation:
# Positive = slower (regression)
# Negative = faster (unlikely to be bad)
```

**When it fires**: Infrastructure or model performance has degraded

---

## 🎯 Decision Logic

### Health States

| State | Meaning | Typical Action |
|-------|---------|----------------|
| `HEALTHY` | Model performing as expected | None |
| `SUSPICIOUS` | Some drift detected, not conclusive | Alert, investigate |
| `UNSTABLE` | Clear degradation, action needed | Rollback |

### Decision Rules

The decision engine uses **multi-signal logic**. This is critical: **drift alone is NOT enough**.

```python
# Pseudo-code for decision logic

if business_proxy_score >= 0.9:
    # Business is fine, ignore drift signals
    return HEALTHY

if business_proxy_score < 0.1:
    # Business is suffering, even without drift
    return UNSTABLE + ROLLBACK

drift_count = count_breached_thresholds()

if drift_count >= 3:
    # Strong evidence: multiple independent signals
    return UNSTABLE + ROLLBACK

if drift_count == 2:
    if has_embedding_drift AND has_confidence_drift:
        # Particularly concerning combination
        return UNSTABLE + ROLLBACK
    else:
        # Investigate but don't act yet
        return SUSPICIOUS + ALERT

if drift_count == 1:
    # Could be noise or early warning
    return SUSPICIOUS + ALERT

# No signals
return HEALTHY
```

### Why Multi-Signal?

| Scenario | Single-Signal Response | Multi-Signal Response |
|----------|------------------------|----------------------|
| Random noise in one metric | ❌ False alarm rollback | ✅ Ignore (HEALTHY) |
| Seasonal traffic change | ❌ Unnecessary rollback | ✅ Alert only (SUSPICIOUS) |
| Actual model degradation | ✅ Correct rollback | ✅ Correct rollback |

---

## ⚙️ Configuration Reference

```python
from prediction_guard.types import GuardConfig, RollbackMechanism

config = GuardConfig(
    # === Drift Thresholds ===
    # Tune these based on your model's sensitivity
    feature_drift_threshold=0.15,       # KS statistic threshold
    embedding_drift_threshold=0.20,     # Cosine distance threshold
    prediction_drift_threshold=0.10,    # PSI threshold
    confidence_entropy_threshold=0.25,  # Relative entropy change
    latency_p99_threshold_ms=100.0,     # Absolute P99 threshold
    
    # === Analysis Windows ===
    analysis_window_minutes=60,         # How much recent data to analyze
    baseline_window_days=7,             # How much data for baseline
    min_samples_for_analysis=100,       # Minimum events for valid analysis
    
    # === Rollback Settings ===
    auto_rollback_enabled=False,        # CRITICAL: Off by default
    rollback_cooldown_minutes=30,       # Minimum time between rollbacks
    rollback_mechanism=RollbackMechanism.CONFIG_FILE,
    
    # === Paths ===
    log_directory="./logs",
    baseline_directory="./baselines",
    incident_directory="./incidents",
    
    # === Model Versions ===
    current_model_version="v2.0",
    fallback_model_version="v1.9",
    
    # === Business Proxy (Optional) ===
    business_proxy_enabled=False,
    business_proxy_threshold=0.10,
    business_proxy_overrides_drift=True,  # Business trumps drift
)
```

### Configuration File (JSON)

```json
{
  "feature_drift_threshold": 0.15,
  "embedding_drift_threshold": 0.20,
  "prediction_drift_threshold": 0.10,
  "confidence_entropy_threshold": 0.25,
  "latency_p99_threshold_ms": 100.0,
  "analysis_window_minutes": 60,
  "baseline_window_days": 7,
  "min_samples_for_analysis": 100,
  "auto_rollback_enabled": false,
  "rollback_cooldown_minutes": 30,
  "rollback_mechanism": "config_file",
  "log_directory": "./logs",
  "baseline_directory": "./baselines",
  "incident_directory": "./incidents",
  "current_model_version": "v2.0",
  "fallback_model_version": "v1.9"
}
```

### Environment Variable

```bash
export PREDICTION_GUARD_CONFIG=/path/to/config.json
```

---

## 🖥️ CLI Reference

### Initialize

```bash
# Create default configuration file
prediction-guard init
```

### Analyze

```bash
# Run drift analysis
prediction-guard analyze --model v2.0 --window 60

# Output as JSON
prediction-guard analyze --model v2.0 --json
```

### Decide

```bash
# Run analysis and make decision
prediction-guard decide --model v2.0

# With business proxy score
prediction-guard decide --model v2.0 --business-score 0.95

# JSON output
prediction-guard decide --json
```

### Run Full Pipeline

```bash
# Analyze, decide, (optionally) act
prediction-guard run --model v2.0

# Actually execute rollback if recommended
prediction-guard run --model v2.0 --execute
```

### Baseline Management

```bash
# Create baseline from last 7 days of data
prediction-guard baseline create --model v2.0 --days 7

# List available baselines
prediction-guard baseline list

# Show baseline details
prediction-guard baseline show --model v2.0

# Delete baseline
prediction-guard baseline delete --model v2.0
```

### Status

```bash
# Show system status
prediction-guard status
```

Output:
```json
{
  "current_model_version": "v2.0",
  "fallback_model_version": "v1.9",
  "auto_rollback_enabled": false,
  "has_baseline": true,
  "available_baselines": ["v1.9", "v2.0"],
  "recent_incidents": [],
  "cooldown_active": false,
  "cooldown_remaining_seconds": 0.0
}
```

### Incidents

```bash
# List recent incidents
prediction-guard incidents --limit 10

# Filter by model
prediction-guard incidents --model v2.0
```

---

## 🐍 Python API Reference

### PredictionGuard (Main Orchestrator)

```python
from prediction_guard import PredictionGuard

guard = PredictionGuard()  # Loads config from file

# Run analysis
analysis = guard.analyze(model_version="v2.0")

# Make decision
decision = guard.decide(analysis, business_proxy_score=0.95)

# Or do both at once
decision = guard.analyze_and_decide(model_version="v2.0")

# Execute action
if decision.recommended_action == ActionType.ROLLBACK:
    action = guard.execute_action(decision, force=False)

# Full pipeline
result = guard.run_pipeline(
    model_version="v2.0",
    auto_execute=False,  # Don't auto-execute
)

# Get system status
status = guard.get_status()
```

### PredictionInterceptor (Middleware)

```python
from prediction_guard.middleware import PredictionInterceptor

interceptor = PredictionInterceptor(config)

# Context manager style (recommended)
with interceptor.intercept(input_data, {"region": "us-east"}) as ctx:
    result = model.predict(input_data)
    ctx.set_result(
        prediction=result.prediction,
        confidence=result.confidence,
        probabilities=result.probabilities,
        embedding=result.embedding,
    )

# Direct logging style
request_id = interceptor.log_prediction(
    input_data=input_data,
    prediction="positive",
    confidence=0.92,
    probabilities=[0.92, 0.08],
    embedding=[0.1, 0.2, 0.3],
    request_context={"region": "us-east"},
    latency_ms=45.2,
)

# Always close when done
interceptor.close()
```

### HealthDecision (Output)

```python
decision = guard.analyze_and_decide()

print(decision.model_version)     # "v2.0"
print(decision.state)             # ModelHealthState.UNSTABLE
print(decision.reasons)           # ["embedding_drift_high", "confidence_entropy_spike"]
print(decision.recommended_action) # ActionType.ROLLBACK
print(decision.confidence)        # 0.85
print(decision.analysis_summary)  # {"feature_drift_score": 0.12, ...}

# Serialize
data = decision.to_dict()
```

---

## 🔄 Rollback Mechanisms

### 1. Config File (Default)

Updates `prediction_guard_config.json`:

```json
{
  "current_model_version": "v1.9",
  "_rollback_at": "2024-01-15T10:30:00",
  "_rollback_from": "v2.0"
}
```

### 2. Environment Variable

Sets:
```bash
MODEL_VERSION=v1.9
MODEL_ROLLBACK_AT=2024-01-15T10:30:00
```

### 3. Model Alias File

Creates/updates `model_alias.json`:
```json
{
  "current_alias": "v1.9",
  "previous_alias": "v2.0",
  "switched_at": "2024-01-15T10:30:00"
}
```

### 4. Feature Flag File

Creates/updates `feature_flags.json`:
```json
{
  "active_model_version": "v1.9",
  "model_rollback_active": true,
  "rollback_at": "2024-01-15T10:30:00"
}
```

### Custom Rollback Handler

For custom integrations (e.g., Kubernetes, service mesh):

```python
# Extend ActionExecutor with custom handler
from prediction_guard.action import ActionExecutor

class CustomExecutor(ActionExecutor):
    def _rollback_custom(self, action):
        # Your custom rollback logic
        # e.g., update Kubernetes ConfigMap
        # e.g., call service mesh API
        pass
```

---

## 📚 Learning Loop

After each incident, Prediction Guard saves a snapshot for post-mortem analysis:

```python
from prediction_guard.incident import IncidentManager

manager = IncidentManager(config)

# Record an incident (automatic when state != HEALTHY)
incident_id = manager.record_incident(decision, analysis, action)

# Add human notes after investigation
manager.add_resolution_notes(
    incident_id=incident_id,
    notes="False positive. Traffic spike from marketing campaign.",
    threshold_adjustments={
        "feature_drift_threshold": 0.20,  # Should be higher
    }
)

# Get aggregated threshold recommendations
recommendations = manager.get_threshold_recommendations(model_version="v2.0")
# {"feature_drift_threshold": 0.18, ...}
```

### Incident Snapshot Structure

```json
{
  "incident_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_version": "v2.0",
  "detected_at": "2024-01-15T10:30:00",
  "decision": {
    "state": "unstable",
    "reasons": ["embedding_drift_high", "confidence_entropy_spike"],
    "recommended_action": "rollback"
  },
  "analysis": {
    "sample_count": 1523,
    "feature_drift_score": 0.12,
    "embedding_drift_score": 0.45,
    "...": "..."
  },
  "action_taken": {
    "success": true,
    "mechanism": "config_file",
    "from_version": "v2.0",
    "to_version": "v1.9"
  },
  "resolution_notes": "Investigating root cause...",
  "threshold_adjustments": {}
}
```

**Important**: No auto-learning in v1. Human-in-the-loop tuning only.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_decision_engine.py -v

# With coverage
pytest tests/ --cov=prediction_guard --cov-report=html
```

### Test Structure

```
tests/
├── test_drift_detector.py    # Statistical test verification
├── test_decision_engine.py   # Decision logic validation
└── test_helpers.py           # Utility function tests
```

---

## 📁 File Structure

```
prediction_guard/
├── __init__.py                 # Package init, version info
├── types.py                    # All types: enums, dataclasses
│   ├── ModelHealthState        # HEALTHY, SUSPICIOUS, UNSTABLE
│   ├── DriftType               # FEATURE, EMBEDDING, PREDICTION, etc.
│   ├── ActionType              # NONE, ALERT, ROLLBACK
│   ├── PredictionEvent         # Single prediction telemetry
│   ├── DriftMetric             # Single drift measurement
│   ├── AnalysisResult          # Complete analysis output
│   ├── HealthDecision          # Decision with reasons
│   ├── RollbackAction          # Executed rollback record
│   └── GuardConfig             # All configuration options
├── config.py                   # Load/save configuration
├── guard.py                    # Main PredictionGuard orchestrator
├── incident.py                 # Incident snapshots for learning loop
├── cli.py                      # Command-line interface
│
├── logging/
│   ├── __init__.py
│   ├── telemetry_logger.py     # Append-only JSONL logging
│   │   └── TelemetryLogger     # Thread-safe, buffered writes
│   └── log_reader.py           # Time-windowed log reading
│       └── LogReader           # Memory-efficient streaming
│
├── analysis/
│   ├── __init__.py
│   ├── drift_detector.py       # Statistical tests
│   │   └── DriftDetector       # KS, PSI, cosine, entropy
│   ├── baseline_manager.py     # Baseline storage
│   │   └── BaselineManager     # Save/load/compute baselines
│   └── analyzer.py             # Orchestrates analysis
│       └── OfflineAnalyzer     # Reads logs, computes all metrics
│
├── decision/
│   ├── __init__.py
│   └── engine.py               # Decision logic
│       └── DecisionEngine      # Multi-signal evaluation
│
├── action/
│   ├── __init__.py
│   ├── executor.py             # Rollback execution
│   │   └── ActionExecutor      # Multiple mechanisms, logging
│   └── cooldown.py             # Cooldown management
│       └── CooldownManager     # Prevent rollback storms
│
└── middleware/
    ├── __init__.py
    ├── interceptor.py          # FastAPI-compatible middleware
    │   └── PredictionInterceptor
    └── helpers.py              # Utilities
        ├── compute_input_hash()
        ├── compute_embedding_summary()
        └── compute_entropy()
```

---

## 🚫 Non-Goals (Explicitly NOT Built)

| Not Building | Why |
|--------------|-----|
| Dashboards | Use Grafana/Datadog for visualization |
| Real-time streaming | Adds complexity without proportional value |
| Auto-threshold tuning | Requires more data and can be dangerous |
| Perfect thresholds | No such thing—tune based on your domain |
| Deep learning models | Overkill for drift detection |
| Replace observability | Complement, don't replace |

**Prediction Guard does ONE thing**: Detect model failure and decide when to roll back.

---

## 🔧 Production Deployment

### Recommended Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Inference API     │     │   Cron Job          │
│   (FastAPI/Flask)   │     │   (every 15 min)    │
│                     │     │                     │
│   + Interceptor     │     │   prediction-guard  │
│     (logs events)   │     │   run --model v2.0  │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           │ writes                    │ reads
           ▼                           ▼
    ┌──────────────────────────────────────────┐
    │          Shared Filesystem               │
    │          (or S3/GCS bucket)              │
    │                                          │
    │   logs/predictions_2024-01-15.jsonl      │
    │   baselines/baseline_v2.0.json           │
    │   incidents/incident_*.json              │
    └──────────────────────────────────────────┘
```

### Kubernetes CronJob Example

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: prediction-guard-analysis
spec:
  schedule: "*/15 * * * *"  # Every 15 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: guard
            image: your-registry/prediction-guard:latest
            command:
            - prediction-guard
            - run
            - --model
            - v2.0
            - --execute  # Only if auto_rollback_enabled
            volumeMounts:
            - name: logs
              mountPath: /app/logs
          volumes:
          - name: logs
            persistentVolumeClaim:
              claimName: prediction-logs
          restartPolicy: OnFailure
```

### Alerting Integration

```python
# After running pipeline
result = guard.run_pipeline()

if result["decision"]["state"] in ["suspicious", "unstable"]:
    # Send to your alerting system
    send_to_pagerduty(
        severity="critical" if result["decision"]["state"] == "unstable" else "warning",
        summary=f"Model {result['model_version']} is {result['decision']['state']}",
        details=result["decision"],
    )
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with the philosophy that **MLOps should be about decisions, not dashboards**.

Inspired by real-world ML incidents where monitoring showed the problem but didn't tell anyone what to do about it.

---

<div align="center">

**Prediction Guard** — *The smallest system that actually decides and acts.*

</div>
