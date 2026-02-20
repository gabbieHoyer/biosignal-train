# src/biosignals/agent/tools.py
"""
smolagents tool definitions for feedback-driven biosignal experiments.

Design principle:
    The agent interacts through well-defined tools at three capability tiers:

    Tier 1 — Config-aware (read-only):
        - run_training(overrides) -> run_dir
        - read_run_results(run_dir) -> structured metrics
        - compare_runs(run_dirs) -> ranked comparison
        - check_drift() -> drift report
        - get_experiment_history() -> compact history
        - list_available_configs() -> config discovery
        - suggest_search_space(run_dir) -> next-step suggestions

    Tier 2 — Config-generating (write YAML, never Python):
        - create_config_variant(group, base, name, modifications)
        - compose_experiment_config(name, components)

    Tier 3 — Code-generating (write model Python, AST-validated):
        - read_model_source(model_name) -> source code + class info
        - register_generated_model(name, code, constructor_args)

Each tool returns a string (required by smolagents) that the
CodeAgent can parse and reason about.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
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

    If a human-in-the-loop approval hook is active, the human will
    be asked to approve, modify, or reject the experiment before
    it launches. The agent does NOT need to know about this — the
    hook is transparent.

    Args:
        overrides: Space-separated Hydra override string.
            Example: "experiment=galaxyppg_hr_ppg trainer.epochs=10 trainer.lr=0.001"

    Returns:
        JSON string with keys: success, run_dir, returncode, error.
        On success, run_dir points to the Hydra output directory containing
        metrics.jsonl, summary.json, config_resolved.yaml, and checkpoints/.
    """
    # ── Human-in-the-loop approval check ──
    from biosignals.agent.hooks import get_approval_hook

    hook = get_approval_hook()
    if hook is not None:
        decision = hook(overrides, _feedback_store)
        if not decision.approved:
            return json.dumps(
                {
                    "success": False,
                    "run_dir": None,
                    "returncode": -2,
                    "error": f"Experiment rejected by human reviewer: {decision.reason}",
                }
            )
        # Apply modifications if any
        if decision.action == "modify" and decision.modified_overrides:
            log.info("Human modified overrides: %s → %s", overrides, decision.modified_overrides)
            overrides = decision.modified_overrides

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
        return json.dumps(
            {
                "success": False,
                "run_dir": None,
                "returncode": -1,
                "error": "Training timed out after 3600 seconds",
            }
        )

    # Parse run directory from Hydra output
    # Hydra prints the output dir; we can also scan for it
    run_dir = _extract_run_dir(result.stdout, result.stderr)

    if result.returncode != 0:
        error_tail = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
        return json.dumps(
            {
                "success": False,
                "run_dir": run_dir,
                "returncode": result.returncode,
                "error": error_tail,
            }
        )

    # Register with feedback store
    if run_dir:
        try:
            _feedback_store.add_run_dir(run_dir, overrides=override_list)
        except Exception as e:
            log.warning("Failed to parse run artifacts at %s: %s", run_dir, e)

    return json.dumps(
        {
            "success": True,
            "run_dir": run_dir,
            "returncode": 0,
            "error": None,
        }
    )


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
        candidates = sorted(
            outputs_root.rglob("config_resolved.yaml"), key=lambda p: p.stat().st_mtime
        )
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
        lines.append(
            "Final train metrics: "
            + json.dumps({k: round(v, 6) for k, v in last.train.items()}, indent=None)
        )
        if last.val:
            lines.append(
                "Final val metrics:   "
                + json.dumps({k: round(v, 6) for k, v in last.val.items()}, indent=None)
            )

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
            runs.append(
                {
                    "dir": d,
                    "run": run,
                    "value": run.best_monitor_value,
                    "failed": run.failed,
                }
            )
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


# ─────────────────────────────────────────────────
# Config discovery helpers
# ─────────────────────────────────────────────────

# Hydra config root relative to project root.
# From train.py: @hydra.main(config_path="../../../configs", ...)
_CONFIG_ROOT_CANDIDATES = ["configs", "conf", "config"]


def _find_config_root() -> Optional[Path]:
    """
    Locate the Hydra config directory.

    Searches from CWD (expected to be project root) for known config
    directory names. Returns the first match, or None.
    """
    cwd = Path.cwd()
    for name in _CONFIG_ROOT_CANDIDATES:
        candidate = cwd / name
        if candidate.is_dir():
            return candidate
    # Also check if we're inside src/ and need to go up
    for parent in [cwd.parent, cwd.parent.parent]:
        for name in _CONFIG_ROOT_CANDIDATES:
            candidate = parent / name
            if candidate.is_dir():
                return candidate
    return None


def _read_yaml_safe(path: Path) -> Dict[str, Any]:
    """Read a YAML file, returning {} on any error."""
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(str(path))
            return OmegaConf.to_container(cfg, resolve=False) or {}
        except Exception:
            return {}


def _summarize_yaml_config(path: Path, group_name: str) -> str:
    """Extract a one-line summary from a config YAML file."""
    data = _read_yaml_safe(path)
    name = path.stem

    # For experiment configs, show what they override
    if group_name == "experiment":
        defaults = data.get("defaults", [])
        overrides = []
        for d in defaults:
            if isinstance(d, dict):
                for k, v in d.items():
                    if k.startswith("override"):
                        # "override /model: encoder_classifier"
                        overrides.append(f"{k.split('/')[-1]}={v}" if "/" in k else f"{v}")
                    elif k != "_self_":
                        overrides.append(f"{k}={v}")
        if overrides:
            return f"{name}: overrides [{', '.join(overrides)}]"
        return name

    # For model configs, show _target_ if present
    target = data.get("_target_", "")
    if target:
        short_target = target.split(".")[-1]
        # Show key params
        params = {
            k: v
            for k, v in data.items()
            if k != "_target_" and not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }
        param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
        return f"{name}: {short_target}({param_str})" if param_str else f"{name}: {short_target}"

    # For trainer configs, show key fields
    if group_name == "trainer":
        fields = {k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))}
        field_str = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:5])
        return f"{name}: {field_str}" if field_str else name

    return name


# ─────────────────────────────────────────────────
# Tool 7: List Available Configs
# ─────────────────────────────────────────────────


@tool
def list_available_configs() -> str:
    """
    Discover all available Hydra config groups and their options.

    Scans the project's config directory to find available models,
    experiments, trainers, datasets, tasks, and transforms. This
    tells you what overrides are valid for run_training().

    Args:
        (no arguments — scans the project config directory)

    Returns:
        Structured listing of all config groups with their options
        and brief descriptions of each. Use this to discover what
        models, experiments, and settings are available before
        choosing overrides for run_training().
    """
    config_root = _find_config_root()
    if config_root is None:
        return (
            "ERROR: Could not find Hydra config directory. "
            "Looked for: configs/, conf/, config/ relative to CWD and parent dirs. "
            f"CWD is: {Path.cwd()}"
        )

    lines = [
        f"Available Hydra Configs (root: {config_root})",
        "=" * 60,
    ]

    # Prioritize the groups most useful for experiment exploration
    priority_groups = ["model", "experiment", "trainer", "dataset", "task", "transforms"]

    # Discover all config groups (subdirectories)
    all_groups = sorted(
        [d.name for d in config_root.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))],
        key=lambda g: priority_groups.index(g) if g in priority_groups else 100,
    )

    for group in all_groups:
        group_dir = config_root / group
        yamls = sorted(group_dir.rglob("*.yaml"))
        if not yamls:
            continue

        lines.append(f"\n## {group}/ ({len(yamls)} options)")
        lines.append(f"   Override syntax: {group}=<name>")

        for ypath in yamls:
            # Handle nested configs (e.g., transforms/ecg/basic.yaml → ecg/basic)
            rel = ypath.relative_to(group_dir).with_suffix("")
            config_name = str(rel).replace("\\", "/")

            summary = _summarize_yaml_config(ypath, group)
            lines.append(
                f"   - {config_name}: {summary}"
                if summary != config_name
                else f"   - {config_name}"
            )

    lines.append("")
    lines.append("Usage: pass these as overrides to run_training().")
    lines.append(
        "Example: run_training('experiment=galaxyppg_hr_ppg model=transformer1d trainer.lr=0.001')"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────
# Tool 8: Suggest Search Space
# ─────────────────────────────────────────────────


@tool
def suggest_search_space(run_dir: str) -> str:
    """
    Analyze a completed run and suggest what to try next.

    Reads the run's config and metrics, checks what alternative
    configs are available, and proposes concrete override strings
    ranked by likely impact. This is the agent's advisor tool.

    Args:
        run_dir: Absolute path to a completed Hydra run directory.

    Returns:
        Prioritized list of suggested next experiments with concrete
        override strings ready to pass to run_training(). Suggestions
        are based on: available config alternatives, the run's metrics
        and convergence status, and common hyperparameter search patterns.
    """
    # Parse the completed run
    try:
        run = parse_run_dir(run_dir)
    except Exception as e:
        return f"ERROR: Failed to parse run at {run_dir}: {e}"

    # Find available configs
    config_root = _find_config_root()
    available: Dict[str, List[str]] = {}
    if config_root:
        for group_dir in config_root.iterdir():
            if group_dir.is_dir() and not group_dir.name.startswith(("_", ".")):
                group = group_dir.name
                yamls = sorted(group_dir.rglob("*.yaml"))
                available[group] = [
                    str(y.relative_to(group_dir).with_suffix("")).replace("\\", "/") for y in yamls
                ]

    lines = [
        f"Search Space Suggestions for: {run.model_name} (lr={run.lr}, epochs={run.epochs_configured})",
        f"Current best: {run.best_monitor_value:.4f} ({run.monitor_metric} {run.monitor_mode})",
        "=" * 60,
    ]

    suggestions: List[Dict[str, Any]] = []

    # ── 1. Alternative models ──
    if "model" in available:
        current_model_stem = run.model_name.lower()
        alt_models = [
            m for m in available["model"] if m.lower() != current_model_stem and "/" not in m
        ]
        if alt_models:
            for m in alt_models:
                suggestions.append(
                    {
                        "priority": "HIGH" if run.converged else "MEDIUM",
                        "category": "model",
                        "rationale": f"Try different architecture (current: {run.model_name})",
                        "override": f"model={m}",
                    }
                )

    # ── 2. Learning rate variations ──
    current_lr = run.lr
    if current_lr > 0:
        lr_candidates = []
        for factor in [0.1, 0.3, 3.0, 10.0]:
            candidate = current_lr * factor
            if 1e-6 <= candidate <= 0.1:
                lr_candidates.append(candidate)

        for lr in lr_candidates:
            direction = "lower" if lr < current_lr else "higher"
            suggestions.append(
                {
                    "priority": "HIGH",
                    "category": "lr",
                    "rationale": f"Try {direction} learning rate ({current_lr} → {lr})",
                    "override": f"trainer.lr={lr}",
                }
            )

    # ── 3. Epoch count ──
    current_epochs = run.epochs_configured
    if run.converged and current_epochs < 50:
        # Already converged, more epochs probably won't help much
        suggestions.append(
            {
                "priority": "LOW",
                "category": "epochs",
                "rationale": f"Model converged at {current_epochs} epochs — more may not help",
                "override": f"trainer.epochs={current_epochs * 2}",
            }
        )
    elif not run.converged and current_epochs < 100:
        suggestions.append(
            {
                "priority": "HIGH",
                "category": "epochs",
                "rationale": f"Model did NOT converge in {current_epochs} epochs — try more",
                "override": f"trainer.epochs={min(current_epochs * 3, 100)}",
            }
        )

    # ── 4. Alternative experiments ──
    if "experiment" in available:
        alt_experiments = [e for e in available["experiment"] if "/" not in e]  # skip nested
        if alt_experiments:
            lines.append("\nAvailable experiment presets:")
            for exp in alt_experiments:
                exp_path = config_root / "experiment" / f"{exp}.yaml"
                summary = (
                    _summarize_yaml_config(exp_path, "experiment") if exp_path.exists() else exp
                )
                lines.append(f"  experiment={exp} — {summary}")

    # ── 5. Combination suggestions ──
    # If we have both alt models and lr candidates, suggest combos
    if any(s["category"] == "model" for s in suggestions) and current_lr > 0:
        best_alt_model = next((s for s in suggestions if s["category"] == "model"), None)
        if best_alt_model:
            suggestions.append(
                {
                    "priority": "MEDIUM",
                    "category": "combo",
                    "rationale": "Combine new model with tuned lr",
                    "override": f"{best_alt_model['override']} trainer.lr={current_lr}",
                }
            )

    # ── 6. Diagnose issues from metrics ──
    if run.epoch_history:
        last = run.epoch_history[-1]
        if last.val and last.train:
            train_loss = last.train.get("loss", 0)
            val_loss = last.val.get("loss", 0)
            if val_loss > 0 and train_loss > 0:
                gap = (val_loss - train_loss) / train_loss
                if gap > 0.5:
                    lines.append(
                        f"\n⚠ Overfitting detected: train_loss={train_loss:.2f}, val_loss={val_loss:.2f} (gap={gap:.0%})"
                    )
                    lines.append("  Consider: fewer epochs, regularization, or more data")
                elif gap < -0.1:
                    lines.append(
                        "\n⚠ Underfitting: val_loss < train_loss — model may need more capacity or epochs"
                    )

        # Check R² if available
        r2 = last.val.get("r2") if last.val else None
        if r2 is not None and r2 < 0:
            lines.append(f"\n⚠ Negative R² ({r2:.2f}): model is worse than predicting the mean.")
            lines.append(
                "  This strongly suggests underfitting — try more epochs, higher lr, or bigger model."
            )

    # ── Format suggestions ──
    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    suggestions.sort(key=lambda s: priority_order.get(s["priority"], 99))

    lines.append("\nSuggested experiments (ranked by priority):")
    lines.append("-" * 40)
    for i, s in enumerate(suggestions, 1):
        lines.append(f"  {i}. [{s['priority']}] {s['rationale']}")
        lines.append(f"     → run_training(\"{s['override']}\")")

    if not suggestions:
        lines.append("  No obvious improvements to suggest. Consider:")
        lines.append("  - Trying a completely different experiment preset")
        lines.append("  - Adding data augmentation via transforms config")
        lines.append("  - Increasing model capacity")

    # ── Available trainers ──
    if "trainer" in available:
        lines.append(f"\nAvailable trainer configs: {', '.join(available['trainer'])}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════
# TIER 2: Config-Generating Tools
# ═════════════════════════════════════════════════
#
# These tools let the agent WRITE new Hydra YAML configs —
# never Python code. This is the key safety boundary:
#   YAML configs are data, not executable code.
#
# The agent can compose novel configurations from existing
# building blocks, the same way a biologist composes an
# experimental protocol from known reagents + procedures.
# ═════════════════════════════════════════════════


# ─────────────────────────────────────────────────
# Safety validation for config generation
# ─────────────────────────────────────────────────

# Groups the agent is allowed to write configs into.
# Excludes hydra/ (internal), logger/ (credentials), and
# anything that could affect system behavior outside training.
_WRITABLE_GROUPS = {"model", "trainer", "experiment", "transforms", "dataset", "task"}

# Filename safety: only alphanumeric, underscores, hyphens
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

# Agent-generated config marker (appears as YAML comment)
_AGENT_GENERATED_MARKER = "# AUTO-GENERATED by experiment agent — do not edit manually\n"


class ConfigSafetyError(ValueError):
    """Raised when a config generation request violates safety constraints."""

    pass


def _validate_config_name(name: str) -> None:
    """Ensure config name is a safe filesystem identifier."""
    if not _SAFE_NAME_RE.match(name):
        raise ConfigSafetyError(
            f"Invalid config name '{name}'. Must start with a letter, "
            f"contain only [a-zA-Z0-9_-], and be ≤64 chars."
        )


def _validate_group(group: str) -> None:
    """Ensure the config group is in the writable allowlist."""
    if group not in _WRITABLE_GROUPS:
        raise ConfigSafetyError(
            f"Cannot write to config group '{group}'. "
            f"Writable groups: {sorted(_WRITABLE_GROUPS)}"
        )


def _validate_no_code_injection(data: Dict[str, Any], path: str = "") -> None:
    """
    Deep-scan a config dict for potential code injection patterns.

    Blocks _target_ values that don't match known project modules,
    and any values containing executable patterns.
    """
    dangerous_patterns = [
        "import ",
        "exec(",
        "eval(",
        "__import__",
        "os.system",
        "subprocess",
        "shutil.rmtree",
        "open(",
        "Path(",
    ]

    for key, value in data.items():
        full_key = f"{path}.{key}" if path else key

        if isinstance(value, str):
            # Check _target_ fields — must be within project namespace
            if key == "_target_":
                if not (
                    value.startswith("biosignals.")
                    or value.startswith("torch.")
                    or value.startswith("torchvision.")
                ):
                    raise ConfigSafetyError(
                        f"Unsafe _target_ at '{full_key}': '{value}'. "
                        f"Must start with biosignals., torch., or torchvision."
                    )
            # Check all string values for executable patterns
            for pattern in dangerous_patterns:
                if pattern in value:
                    raise ConfigSafetyError(
                        f"Potentially dangerous value at '{full_key}': " f"contains '{pattern}'"
                    )

        elif isinstance(value, dict):
            _validate_no_code_injection(value, full_key)

        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _validate_no_code_injection(item, f"{full_key}[{i}]")
                elif isinstance(item, str):
                    for pattern in dangerous_patterns:
                        if pattern in item:
                            raise ConfigSafetyError(
                                f"Potentially dangerous value at '{full_key}[{i}]': "
                                f"contains '{pattern}'"
                            )


def _write_yaml_config(path: Path, data: Dict[str, Any], comment: str = "") -> None:
    """Write a dict as YAML to disk with safety marker."""
    try:
        import yaml

        content = _AGENT_GENERATED_MARKER
        if comment:
            content += f"# {comment}\n"
        content += yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except ImportError:
        # Fallback: manual YAML generation for simple configs
        content = _AGENT_GENERATED_MARKER
        if comment:
            content += f"# {comment}\n"
        content += _dict_to_yaml(data, indent=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _dict_to_yaml(data: Dict[str, Any], indent: int = 0) -> str:
    """Simple YAML serializer for when PyYAML isn't available."""
    lines = []
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_dict_to_yaml(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    # Inline dict in list — used for Hydra defaults
                    for k, v in item.items():
                        lines.append(f"{prefix}  - {k}: {v}")
                else:
                    lines.append(f"{prefix}  - {item}")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines) + "\n" if lines else ""


def _parse_modifications(modifications: str) -> Dict[str, Any]:
    """
    Parse a modification string into a nested dict.

    Supports dotted paths and basic YAML types:
        "n_heads=8 d_model=256 dropout=0.1"
        "model.in_channels=1 trainer.lr=0.001"
    """
    result: Dict[str, Any] = {}
    if not modifications.strip():
        return result

    for token in modifications.strip().split():
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)

        # Parse value type
        value: Any
        if raw_value.lower() in ("true", "yes"):
            value = True
        elif raw_value.lower() in ("false", "no"):
            value = False
        elif raw_value.lower() in ("null", "none", "~"):
            value = None
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        # Handle dotted paths: "model.in_channels=1" → {"model": {"in_channels": 1}}
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    return result


# ─────────────────────────────────────────────────
# Tool 9: Create Config Variant
# ─────────────────────────────────────────────────


@tool
def create_config_variant(group: str, base_config: str, new_name: str, modifications: str) -> str:
    """
    Create a new Hydra config by cloning an existing one and applying modifications.

    This is the core config-generation tool. It reads an existing config
    (e.g., model/transformer1d.yaml), applies parameter changes, and
    writes a new config file that can be used with run_training().

    SAFETY: The _target_ field (which Python class to instantiate) is
    always preserved from the base config — the agent can change parameters
    but cannot change the underlying implementation.

    Args:
        group: Config group to write into. Must be one of:
            model, trainer, experiment, transforms, dataset, task.
        base_config: Name of the existing config to clone.
            Example: "transformer1d" (for model/transformer1d.yaml)
        new_name: Name for the new config file (without .yaml).
            Example: "transformer1d_8head"
            Must start with letter, contain only [a-zA-Z0-9_-], max 64 chars.
        modifications: Space-separated key=value pairs to change.
            Example: "n_heads=8 d_model=256 n_layers=6"
            Supports dotted paths: "encoder.hidden_dim=512"
            Types auto-detected: int, float, bool (true/false), null, string.

    Returns:
        JSON with keys: success, config_path, config_name, override_syntax, error.
        On success, override_syntax shows exactly how to use this config
        with run_training().
    """
    try:
        # ── Validate inputs ──
        _validate_group(group)
        _validate_config_name(new_name)

        config_root = _find_config_root()
        if config_root is None:
            return json.dumps(
                {
                    "success": False,
                    "config_path": None,
                    "config_name": None,
                    "override_syntax": None,
                    "error": "Cannot find Hydra config directory.",
                }
            )

        # ── Read base config ──
        base_path = config_root / group / f"{base_config}.yaml"
        if not base_path.exists():
            # Try nested path (e.g., transforms/ecg/basic)
            base_path = (
                config_root / group / f"{base_config.replace('/', os.sep)}.yaml"
                if os.sep != "/"
                else base_path
            )
            if not base_path.exists():
                available = [
                    str(p.relative_to(config_root / group).with_suffix(""))
                    for p in (config_root / group).rglob("*.yaml")
                ]
                return json.dumps(
                    {
                        "success": False,
                        "config_path": None,
                        "config_name": None,
                        "override_syntax": None,
                        "error": f"Base config '{group}/{base_config}.yaml' not found. "
                        f"Available in {group}/: {available}",
                    }
                )

        base_data = _read_yaml_safe(base_path)
        if not base_data:
            return json.dumps(
                {
                    "success": False,
                    "config_path": None,
                    "config_name": None,
                    "override_syntax": None,
                    "error": f"Failed to parse base config at {base_path}",
                }
            )

        # ── Preserve _target_ (safety boundary) ──
        original_target = base_data.get("_target_")

        # ── Apply modifications ──
        mods = _parse_modifications(modifications)
        if "_target_" in mods:
            return json.dumps(
                {
                    "success": False,
                    "config_path": None,
                    "config_name": None,
                    "override_syntax": None,
                    "error": "Cannot modify _target_ — this would change the Python "
                    "implementation class. Only parameter changes are allowed.",
                }
            )

        # Deep merge modifications into base
        def _deep_merge(base: Dict, updates: Dict) -> Dict:
            merged = dict(base)
            for k, v in updates.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = _deep_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        new_data = _deep_merge(base_data, mods)

        # Restore _target_ if it was in the base (safety enforcement)
        if original_target is not None:
            new_data["_target_"] = original_target

        # ── Safety scan ──
        _validate_no_code_injection(new_data)

        # ── Check for overwrites ──
        new_path = config_root / group / f"{new_name}.yaml"
        overwrite_warning = ""
        if new_path.exists():
            existing = new_path.read_text(encoding="utf-8")
            if _AGENT_GENERATED_MARKER not in existing:
                return json.dumps(
                    {
                        "success": False,
                        "config_path": None,
                        "config_name": None,
                        "override_syntax": None,
                        "error": f"Config '{group}/{new_name}.yaml' already exists and was NOT "
                        f"agent-generated. Refusing to overwrite human-authored config.",
                    }
                )
            overwrite_warning = " (overwrote previous agent-generated version)"

        # ── Write ──
        comment = f"Variant of {group}/{base_config} with: {modifications}"
        _write_yaml_config(new_path, new_data, comment=comment)

        override_syntax = f"{group}={new_name}"
        log.info("Created config variant: %s → %s%s", base_path, new_path, overwrite_warning)

        return json.dumps(
            {
                "success": True,
                "config_path": str(new_path),
                "config_name": new_name,
                "override_syntax": override_syntax,
                "error": None,
                "message": f"Created {group}/{new_name}.yaml based on {base_config}. "
                f"Use with: run_training('{override_syntax} ...'){overwrite_warning}",
            }
        )

    except ConfigSafetyError as e:
        return json.dumps(
            {
                "success": False,
                "config_path": None,
                "config_name": None,
                "override_syntax": None,
                "error": f"SAFETY: {e}",
            }
        )
    except Exception as e:
        log.exception("create_config_variant failed")
        return json.dumps(
            {
                "success": False,
                "config_path": None,
                "config_name": None,
                "override_syntax": None,
                "error": str(e),
            }
        )


# ─────────────────────────────────────────────────
# Tool 10: Compose Experiment Config
# ─────────────────────────────────────────────────


@tool
def compose_experiment_config(name: str, components: str, extra_overrides: str) -> str:
    """
    Compose a new experiment config from existing component configs.

    Experiment configs in Hydra are "recipes" that combine a model,
    dataset, task, trainer, and transforms into a complete setup.
    This tool creates new experiment recipes from existing ingredients.

    This mirrors how biologists compose experimental protocols:
    "Use cell line X with reagent Y under condition Z."

    Args:
        name: Name for the new experiment config (without .yaml).
            Example: "galaxyppg_hr_ppg_transformer"
        components: Space-separated group=config pairs specifying which
            existing configs to compose. Each pair becomes an override
            in the experiment's defaults list.
            Example: "model=transformer1d dataset=galaxyppg task=regression trainer=default transforms=galaxyppg/hr_ppg"
        extra_overrides: Space-separated key=value pairs for additional
            parameter settings applied on top of the composed defaults.
            Example: "model.in_channels=1 model.num_classes=1 data.num_workers=0 trainer.monitor_metric=val/loss trainer.monitor_mode=min"
            Pass empty string "" if no extra overrides needed.

    Returns:
        JSON with keys: success, config_path, config_name, override_syntax, error.
        On success, override_syntax shows how to use this experiment
        with run_training().
    """
    try:
        # ── Validate name ──
        _validate_config_name(name)

        config_root = _find_config_root()
        if config_root is None:
            return json.dumps(
                {
                    "success": False,
                    "config_path": None,
                    "config_name": None,
                    "override_syntax": None,
                    "error": "Cannot find Hydra config directory.",
                }
            )

        # ── Parse components ──
        comp_pairs = {}
        for token in components.strip().split():
            if "=" not in token:
                continue
            group, config_name = token.split("=", 1)
            comp_pairs[group] = config_name

        if not comp_pairs:
            return json.dumps(
                {
                    "success": False,
                    "config_path": None,
                    "config_name": None,
                    "override_syntax": None,
                    "error": "No components specified. Provide group=config pairs, e.g., "
                    "'model=transformer1d dataset=galaxyppg task=regression'",
                }
            )

        # ── Validate each component exists ──
        for group, config_name in comp_pairs.items():
            config_file = config_root / group / f"{config_name.replace('/', os.sep)}.yaml"
            if not config_file.exists():
                available = (
                    [
                        str(p.relative_to(config_root / group).with_suffix(""))
                        for p in (config_root / group).rglob("*.yaml")
                    ]
                    if (config_root / group).is_dir()
                    else []
                )
                return json.dumps(
                    {
                        "success": False,
                        "config_path": None,
                        "config_name": None,
                        "override_syntax": None,
                        "error": f"Component '{group}/{config_name}.yaml' not found. "
                        f"Available in {group}/: {available}",
                    }
                )

        # ── Build experiment config ──
        # Hydra experiment configs use @package _global_ and override defaults
        defaults = []
        for group, config_name in comp_pairs.items():
            defaults.append({f"override /{group}": config_name})
        defaults.append("_self_")

        # Parse extra overrides into nested dict
        extras = _parse_modifications(extra_overrides) if extra_overrides.strip() else {}

        # Safety scan the extras
        if extras:
            _validate_no_code_injection(extras)

        # ── Build YAML content manually for experiment format ──
        # Experiment configs need the @package _global_ directive and
        # specific defaults format that's easier to write directly
        lines = [
            _AGENT_GENERATED_MARKER.rstrip(),
            f"# Experiment: {name}",
            f"# Components: {components}",
            "# @package _global_",
            "",
            "defaults:",
        ]

        for group, config_name in comp_pairs.items():
            lines.append(f"  - override /{group}: {config_name}")
        lines.append("  - _self_")

        # Add extra overrides as top-level YAML
        if extras:
            lines.append("")
            try:
                import yaml

                extras_yaml = yaml.dump(
                    extras, default_flow_style=False, sort_keys=False, allow_unicode=True
                )
                lines.append(extras_yaml.rstrip())
            except ImportError:
                lines.append(_dict_to_yaml(extras).rstrip())

        content = "\n".join(lines) + "\n"

        # ── Check for overwrites ──
        exp_dir = config_root / "experiment"
        exp_dir.mkdir(parents=True, exist_ok=True)
        new_path = exp_dir / f"{name}.yaml"

        overwrite_warning = ""
        if new_path.exists():
            existing = new_path.read_text(encoding="utf-8")
            if _AGENT_GENERATED_MARKER not in existing:
                return json.dumps(
                    {
                        "success": False,
                        "config_path": None,
                        "config_name": None,
                        "override_syntax": None,
                        "error": f"Experiment '{name}' already exists and was NOT "
                        f"agent-generated. Refusing to overwrite.",
                    }
                )
            overwrite_warning = " (overwrote previous agent-generated version)"

        # ── Write ──
        new_path.write_text(content, encoding="utf-8")

        override_syntax = f"experiment={name}"
        log.info("Composed experiment config: %s%s", new_path, overwrite_warning)

        return json.dumps(
            {
                "success": True,
                "config_path": str(new_path),
                "config_name": name,
                "override_syntax": override_syntax,
                "error": None,
                "message": f"Created experiment/{name}.yaml composing [{components}]. "
                f"Use with: run_training('{override_syntax}'){overwrite_warning}",
            }
        )

    except ConfigSafetyError as e:
        return json.dumps(
            {
                "success": False,
                "config_path": None,
                "config_name": None,
                "override_syntax": None,
                "error": f"SAFETY: {e}",
            }
        )
    except Exception as e:
        log.exception("compose_experiment_config failed")
        return json.dumps(
            {
                "success": False,
                "config_path": None,
                "config_name": None,
                "override_syntax": None,
                "error": str(e),
            }
        )


# ═════════════════════════════════════════════════
# TIER 3: Code-Generating Tools
# ═════════════════════════════════════════════════
#
# These tools let the agent READ existing model source code and
# WRITE new model architectures in Python. This crosses the
# code boundary — unlike YAML, Python IS executable — so we
# enforce strict safety via AST analysis:
#
#   1. All generated code is AST-parsed and validated.
#   2. Imports are restricted to an allowlist (torch, numpy, etc).
#   3. Dangerous function calls are blocked (exec, eval, open, etc).
#   4. Generated code lives in an isolated directory:
#      src/biosignals/models/generated/
#   5. The _target_ always points into that directory.
#   6. Bad code → training fails → agent reads error → tries again.
#      The feedback loop IS the validation.
#
# This mirrors the research frontier: the agent can now propose
# novel architectures, test them, and learn from failures.
# ═════════════════════════════════════════════════


# ─────────────────────────────────────────────────
# AST-based code safety validation
# ─────────────────────────────────────────────────

# Top-level modules allowed in generated model code.
# These are the only modules the agent can import.
_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        # PyTorch ecosystem
        "torch",
        "torchvision",
        "torchaudio",
        # Scientific computing
        "numpy",
        "scipy",
        "einops",
        # Python stdlib (safe subset)
        "typing",
        "math",
        "functools",
        "dataclasses",
        "enum",
        "abc",
        "collections",
        "itertools",
        "operator",
        "numbers",
        # Project modules (for reusing components)
        "biosignals",
    }
)

# Functions that are NEVER allowed in generated code.
_BLOCKED_CALLS = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "open",
        "input",
        "print",  # I/O
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",  # reflection (within generated code)
    }
)

# Attribute access patterns that are blocked.
_BLOCKED_ATTR_CHAINS = [
    "os.system",
    "os.popen",
    "os.exec",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.makedirs",
    "os.path",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "shutil.rmtree",
    "shutil.copy",
    "shutil.move",
    "pathlib.Path",
    "socket.socket",
    "http.client",
    "urllib.request",
]

# Marker for agent-generated Python files.
_AGENT_GENERATED_PY_MARKER = (
    "# AUTO-GENERATED by experiment agent — validated via AST safety scan\n"
)

# Generated models directory (relative to project src root).
_GENERATED_MODELS_SUBPATH = Path("biosignals") / "models" / "generated"


class CodeSafetyError(ValueError):
    """Raised when generated code fails safety validation."""

    pass


def _ast_validate_code(code: str) -> Dict[str, Any]:
    """
    Validate generated model code via AST analysis.

    Returns a dict with:
        - class_name: str — the nn.Module subclass found
        - constructor_params: list of (name, default, annotation) tuples
        - has_forward: bool
        - import_modules: list of imported module names

    Raises CodeSafetyError for any violation.
    """
    # Step 1: Parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeSafetyError(f"Syntax error in generated code: {e}") from e

    # Step 2: Validate imports
    imported_modules: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    raise CodeSafetyError(
                        f"Blocked import: '{alias.name}'. "
                        f"Allowed top-level modules: {sorted(_ALLOWED_IMPORT_ROOTS)}"
                    )
                imported_modules.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    raise CodeSafetyError(
                        f"Blocked import: 'from {node.module} import ...'. "
                        f"Allowed top-level modules: {sorted(_ALLOWED_IMPORT_ROOTS)}"
                    )
                imported_modules.append(node.module)

    # Step 3: Validate no dangerous calls
    for node in ast.walk(tree):
        # Direct function calls: exec(), eval(), open(), etc.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
                raise CodeSafetyError(
                    f"Blocked function call: '{func.id}()'. "
                    f"Generated model code cannot use: {sorted(_BLOCKED_CALLS)}"
                )
            # Attribute calls: os.system(), subprocess.run(), etc.
            if isinstance(func, ast.Attribute):
                chain = _resolve_attr_chain(func)
                if chain:
                    for blocked in _BLOCKED_ATTR_CHAINS:
                        if chain.startswith(blocked) or chain == blocked:
                            raise CodeSafetyError(f"Blocked attribute call: '{chain}()'")

        # Global/nonlocal statements (could escape scope)
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise CodeSafetyError("Generated code cannot use 'global' or 'nonlocal' statements.")

    # Step 4: Find nn.Module subclass
    module_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if any base looks like nn.Module or Module
            for base in node.bases:
                base_name = _resolve_attr_chain(base) or (
                    base.id if isinstance(base, ast.Name) else ""
                )
                if "Module" in base_name or base_name.endswith(".Module"):
                    module_classes.append(node)
                    break

    if not module_classes:
        raise CodeSafetyError(
            "Generated code must define at least one class inheriting from "
            "nn.Module (or torch.nn.Module). No such class found."
        )

    if len(module_classes) > 1:
        names = [c.name for c in module_classes]
        log.warning("Multiple nn.Module classes found: %s. Using the first one.", names)

    target_class = module_classes[0]
    class_name = target_class.name

    # Step 5: Check for forward() method
    has_forward = False
    for item in target_class.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "forward":
                has_forward = True
                break

    if not has_forward:
        raise CodeSafetyError(
            f"Class '{class_name}' must define a 'forward()' method. "
            f"This is required for PyTorch nn.Module."
        )

    # Step 6: Extract __init__ constructor parameters
    constructor_params: List[Dict[str, Any]] = []
    for item in target_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            args = item.args
            # Skip 'self'
            param_names = [a.arg for a in args.args[1:]]
            defaults = args.defaults  # right-aligned with params
            n_defaults = len(defaults)
            n_params = len(param_names)

            for i, pname in enumerate(param_names):
                default_idx = i - (n_params - n_defaults)
                default_val = None
                has_default = False
                if default_idx >= 0:
                    has_default = True
                    default_val = _ast_literal_eval_safe(defaults[default_idx])

                # Annotation
                annotation = None
                if i < len(args.args) - 1:  # skip self offset
                    ann_node = args.args[i + 1].annotation
                    if ann_node:
                        annotation = ast.dump(ann_node)

                constructor_params.append(
                    {
                        "name": pname,
                        "has_default": has_default,
                        "default": default_val,
                        "annotation": annotation,
                    }
                )

            # Also handle **kwargs
            if args.kwarg:
                constructor_params.append(
                    {
                        "name": f"**{args.kwarg.arg}",
                        "has_default": False,
                        "default": None,
                        "annotation": None,
                    }
                )
            break

    return {
        "class_name": class_name,
        "constructor_params": constructor_params,
        "has_forward": has_forward,
        "import_modules": imported_modules,
    }


def _resolve_attr_chain(node: ast.expr) -> Optional[str]:
    """Resolve an AST attribute chain like 'os.path.join' to a string."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _ast_literal_eval_safe(node: ast.expr) -> Any:
    """Safely extract a literal value from an AST default parameter."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        # Complex default — return a string representation
        return f"<expr:{ast.dump(node)}>"


def _find_source_root() -> Optional[Path]:
    """Find the project source root (directory containing biosignals/)."""
    cwd = Path.cwd()
    # Check common patterns
    for candidate in [
        cwd / "src",
        cwd / "lib",
        cwd,
        cwd.parent / "src",
    ]:
        if (candidate / "biosignals").is_dir():
            return candidate
    return None


def _resolve_target_to_source(target: str) -> Optional[Path]:
    """
    Resolve a Hydra _target_ string to a Python source file path.

    Example: 'biosignals.models.resnet.ResNet1D'
           → src/biosignals/models/resnet.py
    """
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module_path, _class_name = parts

    # Convert module.path to file/path.py
    rel_path = Path(module_path.replace(".", "/") + ".py")

    src_root = _find_source_root()
    if src_root and (src_root / rel_path).is_file():
        return src_root / rel_path

    # Also try CWD directly
    if (Path.cwd() / rel_path).is_file():
        return Path.cwd() / rel_path

    return None


# ─────────────────────────────────────────────────
# Tool 11: Read Model Source
# ─────────────────────────────────────────────────


@tool
def read_model_source(model_name: str) -> str:
    """
    Read the Python source code of an existing model architecture.

    Use this to understand how a model is implemented before creating
    a modified version with register_generated_model(). Returns the
    full source file, the class name, and constructor parameters.

    Args:
        model_name: Hydra config name for the model.
            Example: "transformer1d", "resnet1d", "encoder_classifier"
            This corresponds to configs/model/{model_name}.yaml

    Returns:
        The model's Python source code, class name, constructor
        signature, and the _target_ path. Returns an error message
        if the model config or source file cannot be found.
    """
    # Step 1: Find and read the model config
    config_root = _find_config_root()
    if config_root is None:
        return "ERROR: Cannot find Hydra config directory."

    config_path = config_root / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        available = (
            [p.stem for p in (config_root / "model").glob("*.yaml")]
            if (config_root / "model").is_dir()
            else []
        )
        return (
            f"ERROR: Model config '{model_name}' not found at {config_path}. "
            f"Available models: {available}"
        )

    config_data = _read_yaml_safe(config_path)
    target = config_data.get("_target_", "")
    if not target:
        return (
            f"ERROR: Model config '{model_name}' has no _target_ field. "
            f"Config contents: {config_data}"
        )

    # Step 2: Resolve _target_ to source file
    source_path = _resolve_target_to_source(target)
    if source_path is None:
        return (
            f"ERROR: Cannot find source file for _target_='{target}'. "
            f"Expected at: src/{target.rsplit('.', 1)[0].replace('.', '/')}.py\n"
            f"Make sure you're running from the project root."
        )

    # Step 3: Read source
    source_code = source_path.read_text(encoding="utf-8")

    # Step 4: Extract class info
    class_name = target.rsplit(".", 1)[-1]

    # Try to find the class in the AST to get constructor params
    constructor_info = ""
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        params = []
                        args = item.args
                        for a in args.args[1:]:  # skip self
                            params.append(a.arg)
                        if args.kwarg:
                            params.append(f"**{args.kwarg.arg}")
                        constructor_info = f"__init__(self, {', '.join(params)})"
                        break
                break
    except SyntaxError:
        constructor_info = "(could not parse constructor)"

    # Step 5: Get config parameters
    config_params = {k: v for k, v in config_data.items() if not k.startswith("_")}

    lines = [
        f"Model: {model_name}",
        f"Target: {target}",
        f"Source: {source_path}",
        f"Class: {class_name}",
        f"Constructor: {constructor_info}",
        f"Config params: {json.dumps(config_params)}",
        "",
        "=" * 60,
        "SOURCE CODE",
        "=" * 60,
        "",
        source_code,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────
# Tool 12: Register Generated Model
# ─────────────────────────────────────────────────


@tool
def register_generated_model(name: str, code: str, constructor_args: str) -> str:
    """
    Validate and register a new model architecture written by the agent.

    This is the code-generation tool. It takes Python source code for a new
    nn.Module, validates it via AST safety analysis, writes it to the
    generated models directory, and creates a matching Hydra config.

    SAFETY ENFORCEMENT:
    - Code is parsed and validated via AST (not executed during registration).
    - Only imports from allowed modules (torch, numpy, biosignals, etc).
    - No file I/O, network, exec/eval, or system calls allowed.
    - Generated files go to an isolated directory: models/generated/
    - Bad code will fail at training time → agent reads error → retries.

    Args:
        name: Name for the new model (used for both .py file and Hydra config).
            Example: "transformer1d_deep"
            Creates: models/generated/transformer1d_deep.py
                     configs/model/transformer1d_deep.yaml
        code: Complete Python source code for the model.
            Must define exactly one class inheriting from nn.Module (or
            torch.nn.Module) with a forward() method. Include all necessary
            imports at the top.
        constructor_args: Space-separated key=value pairs for the Hydra config.
            These become the default parameters in the YAML config.
            Example: "in_channels=1 num_classes=5 d_model=256 n_heads=8"
            Should match the __init__ parameters of your model class.

    Returns:
        JSON with: success, model_path, config_path, override_syntax,
        class_name, constructor_params, error.
    """
    try:
        # ── Validate name ──
        _validate_config_name(name)

        # ── AST safety validation ──
        validation = _ast_validate_code(code)
        class_name = validation["class_name"]

        # ── Locate directories ──
        src_root = _find_source_root()
        if src_root is None:
            return json.dumps(
                {
                    "success": False,
                    "model_path": None,
                    "config_path": None,
                    "override_syntax": None,
                    "class_name": None,
                    "constructor_params": None,
                    "error": "Cannot find project source root (directory containing biosignals/).",
                }
            )

        config_root = _find_config_root()
        if config_root is None:
            return json.dumps(
                {
                    "success": False,
                    "model_path": None,
                    "config_path": None,
                    "override_syntax": None,
                    "class_name": None,
                    "constructor_params": None,
                    "error": "Cannot find Hydra config directory.",
                }
            )

        gen_dir = src_root / _GENERATED_MODELS_SUBPATH
        gen_dir.mkdir(parents=True, exist_ok=True)

        # ── Ensure __init__.py exists ──
        init_path = gen_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text(
                _AGENT_GENERATED_PY_MARKER
                + '"""Auto-generated model architectures created by the experiment agent."""\n',
                encoding="utf-8",
            )

        # ── Check for overwrites ──
        model_path = gen_dir / f"{name}.py"
        overwrite_warning = ""
        if model_path.exists():
            existing = model_path.read_text(encoding="utf-8")
            if _AGENT_GENERATED_PY_MARKER not in existing:
                return json.dumps(
                    {
                        "success": False,
                        "model_path": None,
                        "config_path": None,
                        "override_syntax": None,
                        "class_name": None,
                        "constructor_params": None,
                        "error": f"File '{model_path}' already exists and was NOT "
                        f"agent-generated. Refusing to overwrite.",
                    }
                )
            overwrite_warning = " (overwrote previous version)"

        # ── Write model source ──
        full_code = _AGENT_GENERATED_PY_MARKER + code
        model_path.write_text(full_code, encoding="utf-8")

        # ── Build _target_ path ──
        # e.g., biosignals.models.generated.transformer1d_deep.Transformer1DDeep
        target = f"biosignals.models.generated.{name}.{class_name}"

        # ── Parse constructor args for YAML config ──
        yaml_params = _parse_modifications(constructor_args) if constructor_args.strip() else {}

        # If no constructor args provided, use defaults from AST
        if not yaml_params and validation["constructor_params"]:
            for p in validation["constructor_params"]:
                if p["name"].startswith("**"):
                    continue
                if p["has_default"] and p["default"] is not None:
                    # Only include serializable defaults
                    if isinstance(p["default"], (int, float, bool, str)):
                        yaml_params[p["name"]] = p["default"]

        # ── Write Hydra config ──
        config_data = {"_target_": target}
        config_data.update(yaml_params)

        config_path = config_root / "model" / f"{name}.yaml"
        config_overwrite = ""
        if config_path.exists():
            existing = config_path.read_text(encoding="utf-8")
            if _AGENT_GENERATED_MARKER not in existing:
                return json.dumps(
                    {
                        "success": False,
                        "model_path": str(model_path),
                        "config_path": None,
                        "override_syntax": None,
                        "class_name": class_name,
                        "constructor_params": None,
                        "error": f"Model config '{name}' already exists and was NOT "
                        f"agent-generated. Model .py was written but config "
                        f"was not. Delete or rename the existing config.",
                    }
                )
            config_overwrite = " (config overwrote previous version)"

        comment = f"Generated model: {class_name} in models/generated/{name}.py"
        _write_yaml_config(config_path, config_data, comment=comment)

        override_syntax = f"model={name}"
        log.info(
            "Registered generated model: %s (%s) → %s%s%s",
            name,
            class_name,
            model_path,
            overwrite_warning,
            config_overwrite,
        )

        return json.dumps(
            {
                "success": True,
                "model_path": str(model_path),
                "config_path": str(config_path),
                "override_syntax": override_syntax,
                "class_name": class_name,
                "constructor_params": [
                    p for p in validation["constructor_params"] if not p["name"].startswith("**")
                ],
                "error": None,
                "message": (
                    f"Registered model '{class_name}' at {model_path}. "
                    f"Hydra config at {config_path}. "
                    f"Use with: run_training('{override_syntax} ...'){overwrite_warning}"
                ),
            }
        )

    except (CodeSafetyError, ConfigSafetyError) as e:
        return json.dumps(
            {
                "success": False,
                "model_path": None,
                "config_path": None,
                "override_syntax": None,
                "class_name": None,
                "constructor_params": None,
                "error": f"SAFETY: {e}",
            }
        )
    except Exception as e:
        log.exception("register_generated_model failed")
        return json.dumps(
            {
                "success": False,
                "model_path": None,
                "config_path": None,
                "override_syntax": None,
                "class_name": None,
                "constructor_params": None,
                "error": str(e),
            }
        )
