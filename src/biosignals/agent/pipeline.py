# src/biosignals/agent/pipeline.py
"""
ZenML pipeline for feedback-driven biosignal experiments.

Wraps the smolagents experiment loop into a versioned, reproducible
pipeline with MLflow experiment tracking.

This is the OUTER LOOP: it manages the lifecycle of an entire
experiment campaign (multiple training runs), while the smolagents
agent manages the INNER LOOP (deciding what to try next).

Usage:
    # From CLI
    python -m biosignals.agent.pipeline \\
        --goal "Minimize val/mae on GalaxyPPG HR regression" \\
        --budget 5

    # From Python
    from biosignals.agent.pipeline import feedback_driven_pipeline
    feedback_driven_pipeline(
        goal="Minimize val/mae on GalaxyPPG HR regression",
        budget=5,
    )

Requires:
    pip install zenml mlflow smolagents
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml stack register agent_stack -e mlflow_tracker ... (or use default)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

log = logging.getLogger("biosignals.agent")

# ─────────────────────────────────────────────────
# Lazy imports — ZenML is optional
# ─────────────────────────────────────────────────

try:
    from zenml import pipeline, step
    from zenml.client import Client

    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False

    # Provide no-op decorators so the module is importable without ZenML
    def step(func=None, **kwargs):  # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f

    def pipeline(func=None, **kwargs):  # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f


# ─────────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────────


@step
def define_campaign(goal: str, budget: int, model_id: str) -> Dict[str, Any]:
    """
    Step 1: Define the experiment campaign parameters.

    This step captures the goal, budget, and LLM model choice
    as a versioned ZenML artifact.
    """
    campaign = {
        "goal": goal,
        "budget": budget,
        "model_id": model_id,
    }
    log.info("Campaign defined: %s", json.dumps(campaign, indent=2))
    return campaign


@step
def run_agent_loop(campaign: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Execute the smolagents experiment loop.

    The agent runs training experiments, reads results, checks for
    drift/stagnation, and iteratively improves. This is the core
    feedback-driven component.
    """
    from biosignals.agent.orchestrator import run_experiment_loop
    from biosignals.agent.tools import get_feedback_store

    goal = campaign["goal"]
    budget = campaign["budget"]
    model_id = campaign["model_id"]

    # Run the agent
    agent_summary = run_experiment_loop(
        goal=goal,
        budget=budget,
        model_id=model_id,
        max_steps=budget * 6 + 10,  # ~6 steps per run + overhead
    )

    # Capture feedback store state as artifact
    store = get_feedback_store()
    store_state = store.to_dict()

    return {
        "agent_summary": agent_summary,
        "store_state": store_state,
        "n_runs": store.n_runs,
    }


@step
def evaluate_best(agent_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Evaluate the best run found by the agent.

    Reads the best run's artifacts and produces a final evaluation
    report. In production, this could also register the model in
    MLflow's model registry.
    """
    from biosignals.agent.feedback import parse_run_dir

    store_state = agent_results["store_state"]
    runs = store_state.get("runs", [])

    if not runs:
        return {"status": "no_runs", "best_run": None, "report": "Agent produced no runs."}

    # Find the best run from store state
    drift = store_state.get("drift", {})
    best_run_info = None
    best_value = None

    for r in runs:
        if r.get("failed"):
            continue
        val = r.get("best_value")
        mode = r.get("mode", "min")
        if val is None:
            continue
        if best_value is None:
            best_value = val
            best_run_info = r
        elif mode == "min" and val < best_value:
            best_value = val
            best_run_info = r
        elif mode == "max" and val > best_value:
            best_value = val
            best_run_info = r

    if best_run_info is None:
        return {"status": "no_valid_runs", "best_run": None, "report": "No valid runs found."}

    # Parse the best run for full details
    try:
        best_run = parse_run_dir(best_run_info["run_dir"])
        report = best_run.summary_for_agent()
    except Exception as e:
        report = f"Best run at {best_run_info['run_dir']} (parse error: {e})"

    # Optional: register in MLflow model registry
    _try_mlflow_register(best_run_info)

    return {
        "status": "success",
        "best_run": best_run_info,
        "best_value": best_value,
        "report": report,
        "drift_detected": drift.get("detected", False),
        "recommendation": drift.get("recommendation", "unknown"),
    }


@step
def generate_report(
    campaign: Dict[str, Any],
    agent_results: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> str:
    """
    Step 4: Generate a final campaign report.

    Combines the campaign definition, agent's reasoning, and
    evaluation results into a structured report suitable for
    the portfolio or team review.
    """
    sections = [
        "# Feedback-Driven Experiment Report",
        "",
        "## Campaign",
        f"- **Goal**: {campaign['goal']}",
        f"- **Budget**: {campaign['budget']} runs",
        f"- **Agent model**: {campaign['model_id']}",
        "",
        "## Results",
        f"- **Runs completed**: {agent_results['n_runs']}",
        f"- **Best value**: {evaluation.get('best_value', 'N/A')}",
        f"- **Drift detected**: {evaluation.get('drift_detected', 'N/A')}",
        f"- **Recommendation**: {evaluation.get('recommendation', 'N/A')}",
        "",
        "## Best Run",
        evaluation.get("report", "No valid runs."),
        "",
        "## Agent Summary",
        agent_results.get("agent_summary", "No summary available."),
        "",
        "## All Runs",
    ]

    for i, run in enumerate(agent_results.get("store_state", {}).get("runs", [])):
        status = "FAIL" if run.get("failed") else ("CONV" if run.get("converged") else "OK")
        sections.append(
            f"  {i + 1}. [{status}] {run.get('model', '?')} | "
            f"lr={run.get('lr', '?')} | best={run.get('best_value', '?')} | "
            f"overrides={run.get('overrides', [])}"
        )

    report = "\n".join(sections)
    log.info("Report generated (%d chars)", len(report))
    return report


# ─────────────────────────────────────────────────
# Pipeline Definition
# ─────────────────────────────────────────────────


@pipeline(name="feedback_driven_biosignal", enable_cache=False)
def feedback_driven_pipeline(
    goal: str = "Minimize val/loss on ECG classification with ResNet1D",
    budget: int = 3,
    model_id: str = "anthropic/claude-sonnet-4-20250514",
) -> None:
    """
    Full feedback-driven experiment pipeline.

    Steps:
    1. define_campaign: capture parameters as versioned artifact
    2. run_agent_loop: smolagents agent runs experiments iteratively
    3. evaluate_best: evaluate and optionally register the best model
    4. generate_report: produce human-readable campaign summary
    """
    campaign = define_campaign(goal=goal, budget=budget, model_id=model_id)
    agent_results = run_agent_loop(campaign=campaign)
    evaluation = evaluate_best(agent_results=agent_results)
    generate_report(campaign=campaign, agent_results=agent_results, evaluation=evaluation)


# ─────────────────────────────────────────────────
# MLflow model registry helper
# ─────────────────────────────────────────────────


def _try_mlflow_register(run_info: Dict[str, Any]) -> None:
    """Best-effort model registration in MLflow."""
    try:
        import mlflow
        from pathlib import Path

        run_dir = run_info.get("run_dir")
        if not run_dir:
            return

        ckpt_path = Path(run_dir) / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(run_dir) / "checkpoints" / "last.pt"
        if not ckpt_path.exists():
            return

        mlflow.log_artifact(str(ckpt_path), artifact_path="best_model")
        log.info("Registered best model checkpoint in MLflow: %s", ckpt_path)
    except Exception as e:
        log.debug("MLflow registration skipped: %s", e)


# ─────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run feedback-driven experiment pipeline")
    parser.add_argument("--goal", type=str, required=True, help="Experiment goal in natural language")
    parser.add_argument("--budget", type=int, default=3, help="Max training runs")
    parser.add_argument("--model-id", type=str, default="anthropic/claude-sonnet-4-20250514")
    parser.add_argument("--no-zenml", action="store_true", help="Skip ZenML, run directly")
    args = parser.parse_args()

    if args.no_zenml or not ZENML_AVAILABLE:
        # Run without ZenML pipeline wrapper
        from biosignals.agent.orchestrator import run_experiment_loop

        result = run_experiment_loop(
            goal=args.goal,
            budget=args.budget,
            model_id=args.model_id,
        )
        print(result)
    else:
        # Run as ZenML pipeline
        feedback_driven_pipeline(
            goal=args.goal,
            budget=args.budget,
            model_id=args.model_id,
        )


if __name__ == "__main__":
    main()