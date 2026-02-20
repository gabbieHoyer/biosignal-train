# tests/test_simple_orchestration.py
"""
Integration test for the smolagents experiment orchestrator.
Requires ANTHROPIC_API_KEY set in the environment.

Run from project root:
    PYTHONPATH=$PWD/src python -m pytest tests/test_simple_orchestration.py -v -m integration

Or run directly:
    PYTHONPATH=$PWD/src python tests/test_simple_orchestration.py
"""

import os

import pytest

from biosignals.agent import build_agent, get_feedback_store, run_experiment_loop


@pytest.mark.integration
def test_agent_builds_without_error():
    """Verify the agent object constructs without TypeError."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY not set (integration test).")

    agent = build_agent(
        model_id="anthropic/claude-sonnet-4-20250514",
        max_steps=5,
        verbosity=1,
    )
    assert agent is not None
    assert hasattr(agent, "run")

    # smolagents stores tools as a dict {name: Tool}.
    # It also auto-injects a "final_answer" tool, so we expect
    # our 12 tools + 1 final_answer = 13 total.
    assert isinstance(agent.tools, dict)
    expected_tools = {
        # Tier 1
        "list_available_configs",
        "suggest_search_space",
        "run_training",
        "read_run_results",
        "compare_runs",
        "check_drift",
        "get_experiment_history",
        "read_experiment_config",
        # Tier 2
        "create_config_variant",
        "compose_experiment_config",
        # Tier 3
        "read_model_source",
        "register_generated_model",
        # Auto
        "final_answer",
    }
    assert set(agent.tools.keys()) == expected_tools


@pytest.mark.integration
def test_simple_orchestration_smoke():
    """Smoke test: agent builds, runs 1 experiment, returns a result."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        pytest.skip("ANTHROPIC_API_KEY not set (integration test).")

    result = run_experiment_loop(
        goal=(
            "Minimize val/mae for HR regression on GalaxyPPG. "
            "Start with experiment=galaxyppg_hr_ppg. "
            "For speed, ALWAYS use trainer=fast_dev. "
            "Budget: 1 run. Target: val/mae < 9999 (smoke test)."
        ),
        budget=1,
        model_id="anthropic/claude-sonnet-4-20250514",
        max_steps=10,
    )

    # The agent may return a str OR a dict (via final_answer).
    # Both are valid — smolagents passes through whatever the
    # agent gives to final_answer().
    assert result is not None
    if isinstance(result, str):
        assert len(result) > 0
    elif isinstance(result, dict):
        # Agent returned structured data — even better
        assert len(result) > 0
    else:
        # Convert to string and check it's non-empty
        assert len(str(result)) > 0

    # Verify the FeedbackStore registered the run
    store = get_feedback_store()
    assert store.n_runs >= 1, "FeedbackStore should have at least 1 run"


# ─────────────────────────────────────────────────
# Direct execution (no pytest required)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: Set ANTHROPIC_API_KEY first.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)

    print("=" * 60)
    print("Test 1: Agent builds without error")
    print("=" * 60)
    agent = build_agent(
        model_id="anthropic/claude-sonnet-4-20250514",
        max_steps=5,
        verbosity=1,
    )
    print(f"  Agent created: {type(agent).__name__}")
    # agent.tools is a dict {name_str: Tool}, not a list
    print(f"  Tools: {list(agent.tools.keys())}")
    print(f"  Tool count: {len(agent.tools)} (12 custom + 1 final_answer)")
    print("  PASSED\n")

    print("=" * 60)
    print("Test 2: Smoke test — agent runs 1 experiment")
    print("=" * 60)
    result = run_experiment_loop(
        goal=(
            "Minimize val/mae for HR regression on GalaxyPPG. "
            "Start with experiment=galaxyppg_hr_ppg. "
            "For speed, ALWAYS use trainer=fast_dev. "
            "Budget: 1 run. Target: val/mae < 9999 (smoke test)."
        ),
        budget=1,
        model_id="anthropic/claude-sonnet-4-20250514",
        max_steps=10,
    )
    print(f"  Result type: {type(result).__name__}")
    if isinstance(result, dict):
        print(f"  Keys: {list(result.keys())}")
        if "summary" in result:
            print(f"  Summary: {result['summary'][:200]}")
    else:
        print(f"  Result: {str(result)[:200]}")

    store = get_feedback_store()
    print(f"  Runs in FeedbackStore: {store.n_runs}")
    if store.n_runs > 0:
        best = store.best_run()
        if best:
            print(f"  Best run: {best.best_monitor_value:.4f} ({best.monitor_metric})")
    print("  PASSED\n")

    print("All tests passed.")
