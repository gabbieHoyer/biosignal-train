# src/biosignals/agent/run.py
"""
CLI entrypoint for running experiment campaigns.

Replaces the `python -c "from biosignals.agent import ..."` pattern
with a proper command-line interface.

Usage:
    # Run a named campaign goal
    python -m biosignals.agent.run galaxyppg:hr_baseline
    python -m biosignals.agent.run mitbih:aami3_optimize --budget 5

    # Run with inline goal (backward compatible)
    python -m biosignals.agent.run --goal "Minimize val/mae for HR regression..."

    # List all available campaigns
    python -m biosignals.agent.run --list

    # Interactive approval mode
    python -m biosignals.agent.run mitbih:aami3_baseline --approval terminal

    # Custom model and output directory
    python -m biosignals.agent.run galaxyppg:hr_deep \\
        --model anthropic/claude-sonnet-4-20250514 \\
        --dashboard-dir outputs/campaigns/galaxyppg_deep/
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("biosignals.agent")


def _format_campaign_table(campaigns: list) -> str:
    """Format campaign list as a readable table."""
    if not campaigns:
        return "  (no campaigns found)"

    # Group by dataset
    by_dataset: dict = {}
    for c in campaigns:
        ds = c["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = {"description": c.get("dataset_description", ""), "goals": []}
        by_dataset[ds]["goals"].append(c)

    lines = []
    for ds, info in by_dataset.items():
        lines.append(f"\n  {ds}")
        if info["description"]:
            # Truncate long descriptions
            desc = info["description"].strip().replace("\n", " ")
            if len(desc) > 80:
                desc = desc[:77] + "..."
            lines.append(f"    {desc}")
        lines.append("")

        for g in info["goals"]:
            tags = f"  [{', '.join(g['tags'])}]" if g.get("tags") else ""
            target = ""
            if g.get("target_metric") and g.get("target_value") is not None:
                direction = ">" if g.get("target_direction") == "max" else "<"
                target = f"  target: {g['target_metric']} {direction} {g['target_value']}"

            lines.append(f"    {g['ref']}")
            lines.append(f"      {g.get('description', '')}")
            lines.append(f"      budget: {g['budget']}{target}{tags}")
            lines.append("")

    return "\n".join(lines)


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a feedback-driven experiment campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s galaxyppg:hr_baseline                 # named goal
  %(prog)s mitbih:aami3_optimize --budget 5      # override budget
  %(prog)s --goal "Minimize val/mae on ..."      # inline goal
  %(prog)s --list                                 # show all goals
  %(prog)s mitbih:aami3_baseline --approval terminal  # interactive
""",
    )

    # Positional: campaign ref (optional if --goal or --list)
    parser.add_argument(
        "campaign_ref",
        nargs="?",
        default=None,
        help='Campaign goal reference: "dataset:goal_name" (e.g., mitbih:aami3_baseline)',
    )

    # Inline goal (backward compat)
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Inline goal text (alternative to campaign_ref)",
    )

    # Campaign management
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available campaign goals and exit",
    )
    parser.add_argument(
        "--campaigns-dir",
        type=str,
        default=None,
        help="Path to campaigns/ directory (default: auto-detect)",
    )

    # Run parameters
    parser.add_argument("--budget", type=int, default=None, help="Override run budget")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max agent steps")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="LLM model ID (default: claude-sonnet-4)",
    )
    parser.add_argument(
        "--approval",
        type=str,
        default=None,
        choices=["terminal", "auto", None],
        help="Human-in-the-loop approval mode",
    )

    # Output
    parser.add_argument(
        "--dashboard-dir",
        type=str,
        default=None,
        help="Dashboard output directory (default: auto from campaign ref)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip dashboard generation",
    )

    args = parser.parse_args(argv)

    # ── List mode ──
    if args.list:
        from biosignals.agent.campaigns import list_campaigns

        campaigns = list_campaigns(campaigns_dir=args.campaigns_dir)
        print("\nAvailable campaign goals:")
        print(_format_campaign_table(campaigns))
        return

    # ── Resolve goal ──
    from biosignals.agent.campaigns import load_goal, load_goal_from_inline

    campaign_goal = None

    if args.campaign_ref:
        campaign_goal = load_goal(
            args.campaign_ref,
            campaigns_dir=args.campaigns_dir,
            budget_override=args.budget,
            max_steps_override=args.max_steps,
        )

    elif args.goal:
        campaign_goal = load_goal_from_inline(
            args.goal,
            budget=args.budget or 5,
            max_steps=args.max_steps or 30,
        )

    else:
        parser.error("Provide a campaign ref (e.g., mitbih:aami3_baseline) or --goal '...'")

    # ── Display what we're about to run ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT CAMPAIGN")
    print("=" * 60)
    print(campaign_goal.summary())
    print(f"  Model: {args.model}")
    print(f"  Approval: {args.approval or 'none'}")
    print("-" * 60)

    # ── Resolve dashboard dir ──
    dashboard_dir = args.dashboard_dir
    if dashboard_dir is None and not args.no_dashboard:
        # Auto: outputs/campaigns/{dataset}_{goal_name}_{timestamp}/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_dir = f"outputs/campaigns/{campaign_goal.dataset}_{campaign_goal.goal_name}_{ts}"
        Path(dashboard_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Dashboard: {dashboard_dir}/")

    print("=" * 60 + "\n")

    # ── Run ──
    from biosignals.agent.orchestrator import run_experiment_loop

    run_kwargs = campaign_goal.to_run_kwargs()

    run_experiment_loop(
        **run_kwargs,
        model_id=args.model,
        approval=args.approval,
        dashboard=not args.no_dashboard,
        dashboard_dir=dashboard_dir or ".",
    )

    print("\n" + "=" * 60)
    print("  CAMPAIGN COMPLETE")
    print("=" * 60)
    print(f"  Ref: {campaign_goal.ref}")
    if campaign_goal.target_value is not None:
        print(
            f"  Target: {campaign_goal.target_metric} {campaign_goal.target_direction} {campaign_goal.target_value}"
        )
    if dashboard_dir:
        print(f"  Dashboard: {dashboard_dir}/campaign_dashboard.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
