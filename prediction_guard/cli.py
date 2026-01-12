"""
Command-line interface for Prediction Guard.

Run analysis, create baselines, and manage incidents.
"""

import argparse
import json
import sys

from .config import create_default_config_file
from .guard import PredictionGuard


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prediction Guard - ML Model Failure Detection and Rollback"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run drift analysis")
    analyze_parser.add_argument("--model", "-m", help="Model version to analyze")
    analyze_parser.add_argument("--window", "-w", type=int, help="Analysis window in minutes")
    analyze_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # decide command
    decide_parser = subparsers.add_parser("decide", help="Run analysis and make decision")
    decide_parser.add_argument("--model", "-m", help="Model version")
    decide_parser.add_argument("--business-score", type=float, help="Business proxy score (0-1)")
    decide_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # run command
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--model", "-m", help="Model version")
    run_parser.add_argument("--execute", action="store_true", help="Execute recommended action")
    run_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Manage baselines")
    baseline_parser.add_argument("action", choices=["create", "list", "show", "delete"])
    baseline_parser.add_argument("--model", "-m", help="Model version")
    baseline_parser.add_argument("--days", type=int, default=7, help="Days of data for baseline")

    # status command
    subparsers.add_parser("status", help="Show system status")

    # init command
    subparsers.add_parser("init", help="Create default configuration file")

    # incidents command
    incidents_parser = subparsers.add_parser("incidents", help="List incidents")
    incidents_parser.add_argument("--model", "-m", help="Filter by model")
    incidents_parser.add_argument("--limit", type=int, default=10, help="Max incidents")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "init":
        create_default_config_file()
        return 0

    # Load guard
    try:
        guard = PredictionGuard()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    if args.command == "analyze":
        analysis = guard.analyze(
            model_version=args.model,
            window_minutes=args.window,
        )
        if analysis is None:
            print("Insufficient data for analysis")
            return 1

        if args.json:
            print(json.dumps({
                "sample_count": analysis.sample_count,
                "feature_drift_score": analysis.feature_drift_score,
                "embedding_drift_score": analysis.embedding_drift_score,
                "prediction_drift_score": analysis.prediction_drift_score,
                "confidence_entropy_change": analysis.confidence_entropy_change,
            }, indent=2))
        else:
            print(f"Analysis Results ({analysis.sample_count} samples)")
            print(f"  Feature Drift:    {analysis.feature_drift_score:.4f}")
            print(f"  Embedding Drift:  {analysis.embedding_drift_score:.4f}")
            print(f"  Prediction Drift: {analysis.prediction_drift_score:.4f}")
            print(f"  Entropy Change:   {analysis.confidence_entropy_change:.4f}")

    elif args.command == "decide":
        decision = guard.analyze_and_decide(
            model_version=args.model,
            business_proxy_score=args.business_score,
        )
        if decision is None:
            print("Insufficient data for decision")
            return 1

        if args.json:
            print(json.dumps(decision.to_dict(), indent=2))
        else:
            from .decision import DecisionEngine
            engine = DecisionEngine(guard.config)
            print(engine.explain_decision(decision))

    elif args.command == "run":
        result = guard.run_pipeline(
            model_version=args.model,
            auto_execute=args.execute,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"Error: {result['error']}")
                return 1

            decision = result["decision"]
            print(f"Model: {result['model_version']}")
            print(f"State: {decision['state'].upper()}")
            print(f"Action: {decision['recommended_action']}")
            if result["action"]:
                print(f"Action Executed: {result['action']['success']}")

    elif args.command == "baseline":
        if args.action == "create":
            success = guard.create_baseline(args.model, args.days)
            print("Baseline created" if success else "Failed to create baseline")
        elif args.action == "list":
            baselines = guard.baseline_manager.list_baselines()
            for b in baselines:
                print(f"  - {b}")
        elif args.action == "show":
            baseline = guard.baseline_manager.load_baseline(args.model or guard.config.current_model_version)
            if baseline:
                print(json.dumps(baseline, indent=2))
            else:
                print("Baseline not found")
        elif args.action == "delete":
            if not args.model:
                print("--model required for delete")
                return 1
            deleted = guard.baseline_manager.delete_baseline(args.model)
            print("Deleted" if deleted else "Not found")

    elif args.command == "status":
        status = guard.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "incidents":
        incidents = guard.incident_manager.list_incidents(
            model_version=args.model,
            limit=args.limit,
        )
        for inc in incidents:
            print(f"  {inc['incident_id'][:8]}... | {inc['detected_at'][:10]} | {inc['state']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
