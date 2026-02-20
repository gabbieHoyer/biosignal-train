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
  6. (Optional) Asks human approval before each run
  7. Generates a visual campaign dashboard on completion
  8. Stops when budget is exhausted or goal is met

The agent writes Python code at each step to call tools — it never
generates JSON tool-call blobs. This is smolagents' core advantage.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, LogLevel

from biosignals.agent.hooks import (
    ApprovalHook,
    AutoApproveHook,
    TerminalApprovalHook,
    clear_approval_hook,
    set_approval_hook,
)
from biosignals.agent.tools import (
    check_drift,
    compare_runs,
    compose_experiment_config,
    create_config_variant,
    get_experiment_history,
    get_feedback_store,
    list_available_configs,
    read_experiment_config,
    read_model_source,
    read_run_results,
    register_generated_model,
    reset_feedback_store,
    run_training,
    suggest_search_space,
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

### Discovery & Analysis (Tier 1)
- `list_available_configs()`: **Start here.** Discover all available models, experiments, trainers, datasets.
- `suggest_search_space(run_dir)`: After a run, get prioritized suggestions for what to try next.
- `read_run_results(run_dir)`: Read metrics from a completed run.
- `compare_runs(run_dirs)`: Compare multiple runs side-by-side.
- `check_drift()`: Check if experiments are improving or stagnating.
- `get_experiment_history()`: Review all experiments so far.
- `read_experiment_config(run_dir)`: Read the full config of a past run.

### Config Generation (Tier 2)
- `create_config_variant(group, base_config, new_name, modifications)`: Clone an existing config
  and change specific parameters. Example: create "transformer1d_8head" from "transformer1d" with "n_heads=8".
- `compose_experiment_config(name, components, extra_overrides)`: Compose a new experiment
  recipe from existing model, dataset, task, trainer, and transforms configs.

### Architecture Generation (Tier 3)
- `read_model_source(model_name)`: Read the full Python source code of an existing model.
  Use this to understand the implementation before writing a variant.
- `register_generated_model(name, code, constructor_args)`: Write a new model architecture.
  Provide complete Python code for an nn.Module with a forward() method.
  The code is AST-validated for safety (only torch/numpy/biosignals imports allowed).
  A matching Hydra config is auto-generated.

### Execution
- `run_training(overrides)`: Launch a training run with Hydra config overrides.

## Your strategy

1. **FIRST**: Call `list_available_configs()` to discover what's available.
2. Run the baseline experiment specified in the goal.
3. After each run, call `read_run_results()` then `suggest_search_space()`.
4. Use `check_drift()` after 3+ runs to assess the overall trajectory.

### When to use each tier:
- **Tier 2** (config changes): For hyperparameter tuning (lr, epochs, batch_size),
  or combining existing models with different datasets/trainers. Use when you want
  to change parameters but keep the same model architecture.
- **Tier 3** (code generation): When drift detection suggests the model class itself
  is the bottleneck ("change_model" recommendation), or when you want to try an
  architecture variation that can't be expressed as a config change. Workflow:
  1. Call `read_model_source("base_model")` to read the existing implementation
  2. Modify the architecture (add layers, change attention, etc.)
  3. Call `register_generated_model(name, code, constructor_args)` to register it
  4. Call `run_training("model=new_name ...")` to test it

## Config override syntax

Training is configured via Hydra. Overrides use dotted paths:
- `trainer.lr=0.001` — change learning rate
- `model=transformer1d` — switch model architecture
- `experiment=galaxyppg_hr_ppg` — use a preset experiment config

## Important rules

- ALWAYS call `list_available_configs()` early to know what's available.
- Use `read_model_source()` before writing new model code to understand the interface.
- Generated model code MUST define a class inheriting from nn.Module with forward().
- Generated code can only import: torch, numpy, scipy, einops, typing, math, biosignals.
- If a generated model fails training, read the error and fix the code.
- NEVER modify existing source files. Only create new files via the tools.
- Keep experiment overrides minimal — change one or two things at a time.
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
    max_execution_time: int = 900,
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
        max_execution_time: Max seconds per code execution step.
            Default 900 (15 min) to accommodate training runs.
            The smolagents default of 30s is too short for training.

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
            # Tier 1: Discovery & Analysis
            list_available_configs,
            suggest_search_space,
            read_run_results,
            compare_runs,
            check_drift,
            get_experiment_history,
            read_experiment_config,
            # Tier 2: Config Generation
            create_config_variant,
            compose_experiment_config,
            # Tier 3: Architecture Generation
            read_model_source,
            register_generated_model,
            # Execution
            run_training,
        ],
        model=model,
        max_steps=max_steps,
        verbosity_level=_VERBOSITY_MAP.get(verbosity, LogLevel.INFO),
        instructions=SYSTEM_PROMPT,
        # max_execution_time=max_execution_time,
        executor_type="local",
        executor_kwargs={"timeout_seconds": max_execution_time},  # or None to disable
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
    approval: Union[None, str, ApprovalHook] = None,
    dashboard: bool = True,
    dashboard_dir: str = ".",
) -> str:
    """
    Run a complete feedback-driven experiment loop.

    Args:
        goal: Natural language description of the experiment objective.
        model_id: LLM to use for reasoning.
        budget: Maximum number of training runs the agent may launch.
        max_steps: Maximum agent reasoning steps (includes tool calls + thinking).
        api_key: API key for the model provider.
        approval: Human-in-the-loop approval mode:
            - None: no approval (agent runs freely)
            - "terminal": interactive terminal prompts before each run
            - "auto": auto-approve everything (for testing)
            - ApprovalHook instance: custom hook
        dashboard: Whether to generate an HTML dashboard on completion.
        dashboard_dir: Where to write the dashboard files.

    Returns:
        The agent's final answer (summary of what it tried and concluded).
    """
    # ── Set up approval hook ──
    hook: Optional[ApprovalHook] = None
    if isinstance(approval, ApprovalHook):
        hook = approval
    elif approval == "terminal":
        hook = TerminalApprovalHook()
    elif approval == "auto":
        hook = AutoApproveHook()

    if hook is not None:
        set_approval_hook(hook)
        log.info("Approval hook active: %s", type(hook).__name__)
    else:
        clear_approval_hook()

    # ── Build agent and run ──
    agent = build_agent(model_id=model_id, api_key=api_key, max_steps=max_steps)

    prompt = f"""\
## Experiment Goal

{goal}

## Budget

You may run at most {budget} training experiments. Use them wisely:
- **FIRST**: Call `list_available_configs()` to see what models, experiments, and trainers exist.
- Run the baseline to establish a reference point.
- After each run, call `suggest_search_space(run_dir)` for data-driven next steps.
- Use feedback signals (drift, stagnation) to guide decisions.
- If suggest_search_space recommends changing the model architecture:
  1. Read the current model's source with `read_model_source(model_name)`
  2. Create a modified variant with `register_generated_model(name, code, args)`
  3. Test it with `run_training("model=new_name ...")`
- Compare all runs at the end and report the best configuration.

## Deliverable

After all runs, provide:
1. A ranked list of all experiments with their metrics.
2. The best configuration found and its performance.
3. What you learned about this task/dataset from the experiments.
4. Any generated model architectures and why you designed them that way.
5. Suggestions for further improvement if budget were extended.
"""

    log.info("Starting experiment loop: %s", goal[:100])
    result = agent.run(prompt)
    log.info("Experiment loop complete.")

    # ── Generate dashboard ──
    if dashboard:
        try:
            from biosignals.agent.dashboard import generate_dashboard

            store = get_feedback_store()
            generate_dashboard(
                store,
                approval_hook=hook,
                output_dir=dashboard_dir,
                campaign_goal=goal,
                agent_summary=str(result),
                open_browser=True,
            )
        except Exception as e:
            log.warning("Dashboard generation failed: %s", e)

    # ── Clean up hook ──
    clear_approval_hook()

    return result


# -------------------------------------------------
