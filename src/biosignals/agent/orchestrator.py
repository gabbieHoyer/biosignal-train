# src/biosignals/agent/orchestrator.py
"""
smolagents CodeAgent for feedback-driven experiment orchestration.

Replaces the LangGraph experiment_graph.py with a simpler, more stable
code-first agent that:
  1. Takes a research goal (e.g., "minimize heart rate MAE on GalaxyPPG")
  2. Runs experiments using the training CLI
  3. Reads results from standardized artifacts
  4. Checks for drift/stagnation via the FeedbackStore
  5. Proposes and executes follow-up experiments
  6. Stops when budget is exhausted or goal is met

The agent writes Python code at each step to call tools — it never
generates JSON tool-call blobs. This is smolagents' core advantage.
"""
from __future__ import annotations

import logging
from typing import Optional

from smolagents import CodeAgent, LiteLLMModel, InferenceClientModel, LogLevel

from biosignals.agent.tools import (
    run_training,
    read_run_results,
    compare_runs,
    check_drift,
    get_experiment_history,
    read_experiment_config,
    reset_feedback_store,
)

# Map integer verbosity to smolagents LogLevel enum
_VERBOSITY_MAP = {
    0: LogLevel.ERROR,
    1: LogLevel.INFO,
    2: LogLevel.DEBUG,
}

log = logging.getLogger("biosignals.agent")


# ─────────────────────────────────────────────────
# System prompt for the experiment agent
# ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a biosignal ML experiment agent. Your job is to run training experiments,
analyze results, and iteratively improve model performance toward a stated goal.

## How you work

You have access to tools that interact with a PyTorch training system:
- `run_training(overrides)`: Launch a training run with Hydra config overrides
- `read_run_results(run_dir)`: Read metrics from a completed run
- `compare_runs(run_dirs)`: Compare multiple runs side-by-side
- `check_drift()`: Check if experiments are improving or stagnating
- `get_experiment_history()`: Review all experiments so far
- `read_experiment_config(run_dir)`: Read the full config of a past run

## Your strategy

1. Start with the baseline experiment specified in the goal.
2. After each run, read results and check for drift/stagnation.
3. Based on the feedback signals, decide what to change:
   - If the model is stagnating: try different learning rates, more epochs, or a different model.
   - If performance is degrading: revert to the best config and make smaller changes.
   - If performance is improving: continue in the same direction.
4. Always compare new runs against the current best.
5. Stop when you've exhausted the budget OR the goal metric is reached.

## Config override syntax

Training is configured via Hydra. Overrides use dotted paths:
- `trainer.lr=0.001` — change learning rate
- `trainer.epochs=30` — change epoch count
- `model=transformer1d` — switch model architecture
- `experiment=galaxyppg_hr_ppg` — use a preset experiment config
- `model.in_channels=1 model.num_classes=1` — model-specific params

## Important rules

- NEVER modify training code directly. Only use the provided tools.
- ALWAYS read results after running an experiment before deciding next steps.
- ALWAYS check drift after 3+ runs to assess the overall trajectory.
- Keep experiment overrides minimal — change one or two things at a time.
- Track what you've tried to avoid repeating the same configuration.
"""


# ─────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────


def build_agent(
    model_id: str = "anthropic/claude-sonnet-4-20250514",
    *,
    api_key: Optional[str] = None,
    max_steps: int = 20,
    verbosity: int = 1,
) -> CodeAgent:
    """
    Build a smolagents CodeAgent wired with biosignal experiment tools.

    Args:
        model_id: LiteLLM model identifier. Examples:
            - "anthropic/claude-sonnet-4-20250514" (recommended)
            - "openai/gpt-4o"
            - "huggingface/Qwen/Qwen2.5-72B-Instruct" (via HF Inference)
        api_key: API key for the model provider. If None, reads from
            environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
        max_steps: Maximum agent reasoning steps before forced stop.
        verbosity: 0=silent, 1=steps, 2=debug.

    Returns:
        Configured CodeAgent ready to accept .run() calls.
    """
    # Reset feedback store for a fresh campaign
    reset_feedback_store()

    # Build model
    if model_id.startswith("huggingface/"):
        # Use HF Inference API
        hf_model_id = model_id.replace("huggingface/", "")
        model = InferenceClientModel(model_id=hf_model_id)
    else:
        # Use LiteLLM for OpenAI/Anthropic/etc.
        kwargs = {"model_id": model_id, "temperature": 0.2}
        if api_key:
            kwargs["api_key"] = api_key
        model = LiteLLMModel(**kwargs)

    agent = CodeAgent(
        tools=[
            run_training,
            read_run_results,
            compare_runs,
            check_drift,
            get_experiment_history,
            read_experiment_config,
        ],
        model=model,
        max_steps=max_steps,
        verbosity_level=_VERBOSITY_MAP.get(verbosity, LogLevel.INFO),
        instructions=SYSTEM_PROMPT,
    )

    log.info("Built experiment agent with model=%s, max_steps=%d", model_id, max_steps)
    return agent


# ─────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────


def run_experiment_loop(
    goal: str,
    *,
    model_id: str = "anthropic/claude-sonnet-4-20250514",
    budget: int = 5,
    max_steps: int = 30,
    api_key: Optional[str] = None,
) -> str:
    """
    Run a complete feedback-driven experiment loop.

    Args:
        goal: Natural language description of the experiment objective.
            Example: "Minimize val/mae for heart rate regression on GalaxyPPG
            using PPG signals. Start with experiment=galaxyppg_hr_ppg as baseline.
            Budget: 5 runs. Target: val/mae < 5.0"
        model_id: LLM to use for reasoning.
        budget: Maximum number of training runs the agent may launch.
        max_steps: Maximum agent reasoning steps (includes tool calls + thinking).
        api_key: API key for the model provider.

    Returns:
        The agent's final answer (summary of what it tried and concluded).
    """
    agent = build_agent(model_id=model_id, api_key=api_key, max_steps=max_steps)

    prompt = f"""\
## Experiment Goal

{goal}

## Budget

You may run at most {budget} training experiments. Use them wisely:
- Start with the baseline to establish a reference point.
- Use feedback signals (drift, stagnation) to guide decisions.
- Compare all runs at the end and report the best configuration.

## Deliverable

After all runs, provide:
1. A ranked list of all experiments with their metrics.
2. The best configuration found and its performance.
3. What you learned about this task/dataset from the experiments.
4. Suggestions for further improvement if budget were extended.
"""

    log.info("Starting experiment loop: %s", goal[:100])
    result = agent.run(prompt)
    log.info("Experiment loop complete.")
    return result


# --------------------------------------------------------------------
