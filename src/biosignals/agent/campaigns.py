# src/biosignals/agent/campaigns.py
"""
Campaign goal management for experiment orchestration.

Loads, validates, and resolves campaign goals from YAML files.
Each dataset has its own campaign file with named goals that define:
  - Natural language prompt for the agent
  - Budget, target metric, baseline experiment
  - Tags for filtering and categorization

Campaign files live in campaigns/ at the project root (alongside configs/).
This keeps agent prompts version-controlled and separate from model configs.

Design rationale:
  - YAML (not JSON): supports comments and multiline strings
  - Per-dataset files (not monolithic): clean git diffs, no merge conflicts
  - Not Hydra configs: goals are agent prompts, not model parameters
  - Named goals: selectable at runtime via dataset:goal_name syntax

Usage:
    from biosignals.agent.campaigns import load_goal, list_campaigns

    # Load a specific goal
    goal = load_goal("mitbih:aami3_baseline")

    # Load with campaign dir override
    goal = load_goal("galaxyppg:hr_deep", campaigns_dir="path/to/campaigns")

    # List all available campaigns
    for campaign in list_campaigns():
        print(campaign)

    # Use with run_experiment_loop
    from biosignals.agent.orchestrator import run_experiment_loop
    run_experiment_loop(**goal.to_run_kwargs())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger("biosignals.agent")


@dataclass
class CampaignGoal:
    """
    A resolved, ready-to-run campaign goal.

    Contains everything run_experiment_loop() needs, plus metadata
    for tracking and dashboard display.
    """

    # Identity
    dataset: str            # e.g., "galaxyppg", "mitbih"
    goal_name: str          # e.g., "hr_baseline", "aami3_optimize"
    ref: str                # e.g., "galaxyppg:hr_baseline"

    # Agent prompt (the actual goal text sent to the LLM)
    goal: str

    # Run parameters
    budget: int = 5
    max_steps: int = 30
    baseline_experiment: str = ""

    # Target (for dashboard / early stopping)
    target_metric: str = ""
    target_value: Optional[float] = None
    target_direction: str = "min"  # "min" or "max"

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dataset_description: str = ""

    def to_run_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs dict for run_experiment_loop().

        Returns dict with 'goal', 'budget', 'max_steps' —
        the caller can merge with model_id, approval, etc.
        """
        return {
            "goal": self.goal,
            "budget": self.budget,
            "max_steps": self.max_steps,
        }

    def summary(self) -> str:
        """Human-readable summary for display."""
        lines = [
            f"Campaign: {self.ref}",
            f"  {self.description}" if self.description else "",
            f"  Baseline: {self.baseline_experiment}",
            f"  Budget: {self.budget} runs, max {self.max_steps} agent steps",
            f"  Target: {self.target_metric} {self.target_direction} {self.target_value}"
            if self.target_value is not None else "",
            f"  Tags: {', '.join(self.tags)}" if self.tags else "",
        ]
        return "\n".join(line for line in lines if line)


# ─────────────────────────────────────────────────
# Campaign file discovery
# ─────────────────────────────────────────────────


def _find_campaigns_dir(campaigns_dir: Optional[str] = None) -> Path:
    """
    Locate the campaigns/ directory.

    Search order:
    1. Explicit path if provided
    2. campaigns/ relative to CWD
    3. campaigns/ relative to project root (detected by configs/ sibling)
    """
    if campaigns_dir is not None:
        p = Path(campaigns_dir)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Campaigns directory not found: {p}")

    # Try CWD
    cwd_campaigns = Path.cwd() / "campaigns"
    if cwd_campaigns.is_dir():
        return cwd_campaigns

    # Try to find project root by looking for configs/ directory
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "configs").is_dir() and (parent / "campaigns").is_dir():
            return parent / "campaigns"

    raise FileNotFoundError(
        "Cannot find campaigns/ directory. "
        "Create it at your project root (alongside configs/) or pass --campaigns-dir."
    )


def _load_campaign_file(path: Path) -> Dict[str, Any]:
    """Load and validate a single campaign YAML file."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "dataset" not in data:
        raise ValueError(f"Campaign file {path} missing required 'dataset' field")
    if "goals" not in data or not isinstance(data["goals"], dict):
        raise ValueError(f"Campaign file {path} missing 'goals' dict")

    return data


# ─────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────


def list_campaigns(
    campaigns_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all available campaign goals across all datasets.

    Returns list of dicts with keys: ref, dataset, goal_name, description, tags, budget.
    """
    camp_dir = _find_campaigns_dir(campaigns_dir)
    results = []

    for yaml_path in sorted(camp_dir.glob("*.yaml")):
        try:
            data = _load_campaign_file(yaml_path)
        except (ValueError, Exception) as e:
            log.warning("Skipping %s: %s", yaml_path.name, e)
            continue

        dataset = data["dataset"]
        dataset_desc = data.get("description", "")

        for goal_name, goal_data in data["goals"].items():
            results.append({
                "ref": f"{dataset}:{goal_name}",
                "dataset": dataset,
                "goal_name": goal_name,
                "description": goal_data.get("description", ""),
                "tags": goal_data.get("tags", []),
                "budget": goal_data.get("budget", 5),
                "target_metric": goal_data.get("target_metric", ""),
                "target_direction": goal_data.get("target_direction", ""),
                "target_value": goal_data.get("target_value"),
                "dataset_description": dataset_desc,
            })

    return results


def load_goal(
    ref: str,
    *,
    campaigns_dir: Optional[str] = None,
    budget_override: Optional[int] = None,
    max_steps_override: Optional[int] = None,
) -> CampaignGoal:
    """
    Load a campaign goal by reference string.

    Args:
        ref: Goal reference in "dataset:goal_name" format.
            Examples: "galaxyppg:hr_baseline", "mitbih:aami3_optimize"
        campaigns_dir: Override campaigns directory path.
        budget_override: Override the goal's default budget.
        max_steps_override: Override the goal's default max_steps.

    Returns:
        CampaignGoal ready for run_experiment_loop().

    Raises:
        FileNotFoundError: If campaign file doesn't exist.
        KeyError: If goal_name not found in campaign file.
        ValueError: If ref format is invalid.
    """
    # Parse ref
    if ":" not in ref:
        raise ValueError(
            f"Invalid campaign ref '{ref}'. "
            f"Expected format: 'dataset:goal_name' (e.g., 'mitbih:aami3_baseline'). "
            f"Use --list to see available campaigns."
        )

    dataset, goal_name = ref.split(":", 1)

    # Find and load campaign file
    camp_dir = _find_campaigns_dir(campaigns_dir)
    yaml_path = camp_dir / f"{dataset}.yaml"

    if not yaml_path.exists():
        available = [p.stem for p in camp_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"No campaign file for dataset '{dataset}'. "
            f"Expected: {yaml_path}\n"
            f"Available datasets: {available}"
        )

    data = _load_campaign_file(yaml_path)

    # Resolve goal
    goals = data["goals"]
    if goal_name not in goals:
        available_goals = list(goals.keys())
        raise KeyError(
            f"Goal '{goal_name}' not found in {dataset} campaign. "
            f"Available goals: {available_goals}"
        )

    g = goals[goal_name]
    goal_text = g.get("goal", "")
    if not goal_text.strip():
        raise ValueError(f"Goal '{ref}' has empty goal text")

    # Build CampaignGoal
    return CampaignGoal(
        dataset=dataset,
        goal_name=goal_name,
        ref=ref,
        goal=goal_text.strip(),
        budget=budget_override or g.get("budget", 5),
        max_steps=max_steps_override or g.get("max_steps", 30),
        baseline_experiment=g.get("baseline_experiment", ""),
        target_metric=g.get("target_metric", ""),
        target_value=g.get("target_value"),
        target_direction=g.get("target_direction", "min"),
        description=g.get("description", ""),
        tags=g.get("tags", []),
        dataset_description=data.get("description", ""),
    )


def load_goal_from_inline(
    goal_text: str,
    *,
    budget: int = 5,
    max_steps: int = 30,
) -> CampaignGoal:
    """
    Create a CampaignGoal from an inline goal string.

    For backward compatibility with the `python -c` approach.
    """
    return CampaignGoal(
        dataset="inline",
        goal_name="custom",
        ref="inline:custom",
        goal=goal_text.strip(),
        budget=budget,
        max_steps=max_steps,
    )