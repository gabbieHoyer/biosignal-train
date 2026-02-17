# src/biosignals/agent/tools.py
"""
smolagents tool definitions for feedback-driven biosignal experiments.

Design principle (carried forward from original):
    The agent NEVER touches raw training internals.
    It only interacts through well-defined tools:
        - run_training(overrides) -> run_dir
        - read_run_results(run_dir) -> structured metrics
        - compare_runs(run_dirs) -> ranked comparison
        - check_drift() -> drift report
        - get_experiment_history() -> compact history for reasoning

Each tool returns a string (required by smolagents) that the
CodeAgent can parse and reason about.
"""
from __future__ import annotations

import json
import subprocess
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from smolagents import tool

from biosignals.agent.feedback import FeedbackStore, parse_run_dir

log = logging.getLogger("biosignals.agent")

# ─────────────────────────────────────────────────
# Module-level feedback store (shared across tools)
# ─────────────────────────────────────────────────
# The agent's tools share a single FeedbackStore instance.
# This accumulates across the agent's run loop.
_feedback_store = FeedbackStore()


def get_feedback_store() -> FeedbackStore:
    """Access the shared FeedbackStore (for orchestrator/pipeline use)."""
    return _feedback_store


def reset_feedback_store(**kwargs: Any) -> None:
    """Reset the store (e.g., at start of a new experiment campaign)."""
    global _feedback_store
    _feedback_store = FeedbackStore(**kwargs)


# ─────────────────────────────────────────────────
# Tool 1: Run Training
# ─────────────────────────────────────────────────


@tool
def run_training(overrides: str) -> str:
    """
    Launch a biosignal training run with Hydra config overrides.

    This calls the existing training CLI as a subprocess, ensuring
    the agent never touches training internals directly.

    Args:
        overrides: Space-separated Hydra override string.
            Example: "experiment=galaxyppg_hr_ppg trainer.epochs=10 trainer.lr=0.001"

    Returns:
        JSON string with keys: success, run_dir, returncode, error.
        On success, run_dir points to the Hydra output directory containing
        metrics.jsonl, summary.json, config_resolved.yaml, and checkpoints/.
    """
    override_list = overrides.strip().split() if overrides.strip() else []

    cmd = ["python", "-m", "biosignals.cli.train"] + override_list

    log.info("run_training: launching %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,  # 1 hour max per run
        )
    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "run_dir": None,
            "returncode": -1,
            "error": "Training timed out after 3600 seconds",
        })

    # Parse run directory from Hydra output
    # Hydra prints the output dir; we can also scan for it
    run_dir = _extract_run_dir(result.stdout, result.stderr)

    if result.returncode != 0:
        error_tail = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
        return json.dumps({
            "success": False,
            "run_dir": run_dir,
            "returncode": result.returncode,
            "error": error_tail,
        })

    # Register with feedback store
    if run_dir:
        try:
            _feedback_store.add_run_dir(run_dir, overrides=override_list)
        except Exception as e:
            log.warning("Failed to parse run artifacts at %s: %s", run_dir, e)

    return json.dumps({
        "success": True,
        "run_dir": run_dir,
        "returncode": 0,
        "error": None,
    })


def _extract_run_dir(stdout: str, stderr: str) -> Optional[str]:
    """
    Extract the Hydra run directory from training output.

    Looks for the 'Run dir: /path/...' log line produced by train.py,
    or falls back to scanning for the outputs/ directory pattern.
    """
    for text in [stdout, stderr]:
        for line in text.splitlines():
            if "Run dir:" in line:
                # Format: "... Run dir: /absolute/path/to/outputs/..."
                parts = line.split("Run dir:")
                if len(parts) >= 2:
                    candidate = parts[-1].strip()
                    if Path(candidate).is_dir():
                        return candidate

    # Fallback: find the most recent outputs/ subdirectory
    # This works because Hydra creates timestamped directories
    outputs_root = Path("outputs")
    if outputs_root.is_dir():
        candidates = sorted(outputs_root.rglob("config_resolved.yaml"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return str(candidates[-1].parent)

    return None


# ─────────────────────────────────────────────────
# Tool 2: Read Run Results
# ─────────────────────────────────────────────────


@tool
def read_run_results(run_dir: str) -> str:
    """
    Read and summarize results from a completed training run.

    Parses the standardized artifacts (metrics.jsonl, summary.json,
    config_resolved.yaml) from a Hydra run directory.

    Args:
        run_dir: Absolute path to a Hydra run output directory.

    Returns:
        Human-readable summary including: task, model, dataset, learning rate,
        epochs completed, best metric value, convergence status, and the
        full epoch-by-epoch training curve for the monitored metric.
    """
    try:
        run = parse_run_dir(run_dir)
    except Exception as e:
        return f"ERROR: Failed to parse run at {run_dir}: {e}"

    # Build detailed summary
    lines = [run.summary_for_agent(), ""]

    # Training curve
    curve = run.training_curve()
    if curve:
        lines.append("Training curve (epoch -> monitor value):")
        for epoch, val in curve:
            marker = " <-- best" if run.best_epoch is not None and epoch == run.best_epoch else ""
            lines.append(f"  epoch {epoch:3d}: {val:.6f}{marker}")

    # Final train/val metrics
    if run.epoch_history:
        last = run.epoch_history[-1]
        lines.append("")
        lines.append("Final train metrics: " + json.dumps(
            {k: round(v, 6) for k, v in last.train.items()}, indent=None
        ))
        if last.val:
            lines.append("Final val metrics:   " + json.dumps(
                {k: round(v, 6) for k, v in last.val.items()}, indent=None
            ))

    return "\n".join(lines)


# ─────────────────────────────────────────────────
# Tool 3: Compare Runs
# ─────────────────────────────────────────────────


@tool
def compare_runs(run_dirs: str) -> str:
    """
    Compare multiple training runs side-by-side, ranked by performance.

    Args:
        run_dirs: Comma-separated list of run directory paths.
            Example: "/path/to/run1,/path/to/run2,/path/to/run3"

    Returns:
        Ranked comparison table showing each run's config, best metric,
        convergence status, and relative performance.
    """
    dirs = [d.strip() for d in run_dirs.split(",") if d.strip()]

    if not dirs:
        return "ERROR: No run directories provided."

    runs: List[Dict[str, Any]] = []
    for d in dirs:
        try:
            run = parse_run_dir(d)
            runs.append({
                "dir": d,
                "run": run,
                "value": run.best_monitor_value,
                "failed": run.failed,
            })
        except Exception as e:
            runs.append({"dir": d, "run": None, "value": float("nan"), "failed": True})
            log.warning("Failed to parse %s: %s", d, e)

    # Determine sort direction from the first valid run
    valid = [r for r in runs if r["run"] is not None and not r["failed"]]
    if not valid:
        return "ERROR: No valid runs found."

    mode = valid[0]["run"].monitor_mode
    metric_name = valid[0]["run"].monitor_metric

    import math

    def sort_key(r: Dict) -> float:
        v = r["value"]
        if math.isnan(v):
            return float("-inf")
        return v if mode == "max" else -v

    runs.sort(key=sort_key, reverse=True)

    # Build comparison table
    lines = [
        f"Run Comparison ({len(runs)} runs, ranked by {metric_name} {mode})",
        "=" * 70,
    ]

    for rank, r in enumerate(runs, 1):
        if r["run"] is None:
            lines.append(f"  #{rank} PARSE_ERROR: {r['dir']}")
            continue

        run = r["run"]
        status = "FAIL" if run.failed else ("CONV" if run.converged else "OK")
        lines.append(
            f"  #{rank} [{status}] {run.model_name} | lr={run.lr} | "
            f"best={run.best_monitor_value:.6f} @ ep{run.best_epoch} | "
            f"epochs={run.n_epochs_completed}/{run.epochs_configured}"
        )
        if run.overrides:
            lines.append(f"       overrides: {run.overrides}")

    # Delta from best
    if len(valid) >= 2:
        best_val = valid[0]["run"].best_monitor_value
        lines.append("")
        lines.append("Deltas from best:")
        for r in runs[1:]:
            if r["run"] and not r["failed"]:
                delta = r["value"] - best_val
                lines.append(f"  {r['run'].model_name} lr={r['run'].lr}: {delta:+.6f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────
# Tool 4: Check Drift
# ─────────────────────────────────────────────────


@tool
def check_drift() -> str:
    """
    Analyze experiment history for performance drift or stagnation.

    Examines all runs registered in the feedback store and detects:
    - Drift: recent runs performing WORSE than earlier runs
    - Stagnation: no improvement despite configuration changes

    Returns actionable diagnostics with a recommendation:
    "continue", "change_lr", "change_model", or "stop".

    Args:
        (no arguments — operates on the shared experiment history)

    Returns:
        Drift analysis report including trend slope, stagnation status,
        and a specific recommendation for the next action.
    """
    store = _feedback_store

    if store.n_runs == 0:
        return "No runs recorded yet. Run some experiments first."

    report = store.detect_drift()
    return report.summary_for_agent()


# ─────────────────────────────────────────────────
# Tool 5: Get Experiment History
# ─────────────────────────────────────────────────


@tool
def get_experiment_history() -> str:
    """
    Get a compact summary of all experiments run so far.

    This provides the full context the agent needs to decide
    what to try next: every run's config, metrics, and status,
    plus the current drift/stagnation analysis.

    Args:
        (no arguments — reads from the shared experiment history)

    Returns:
        Formatted experiment history showing all runs with their configs,
        metrics, convergence status, and a drift analysis summary.
    """
    return _feedback_store.history_for_agent()


# ─────────────────────────────────────────────────
# Tool 6: Read Config
# ─────────────────────────────────────────────────


@tool
def read_experiment_config(run_dir: str) -> str:
    """
    Read the resolved Hydra config from a completed run.

    Useful for understanding exactly what parameters were used,
    to inform the next experiment's configuration.

    Args:
        run_dir: Absolute path to a Hydra run output directory.

    Returns:
        The full resolved config as YAML text, or an error message.
    """
    config_path = Path(run_dir) / "config_resolved.yaml"
    if not config_path.exists():
        return f"ERROR: No config_resolved.yaml found at {run_dir}"

    return config_path.read_text(encoding="utf-8")