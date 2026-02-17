# src/biosignals/agent/__init__.py
"""
Feedback-driven experiment orchestration for biosignal models.

Stack: smolagents (agent) + MLflow (tracking) + ZenML (pipeline, optional)

Usage (simple):
    from biosignals.agent import run_experiment_loop

    result = run_experiment_loop(
        goal="Minimize val/mae for HR regression on GalaxyPPG. "
             "Start with experiment=galaxyppg_hr_ppg. Target: val/mae < 5.0",
        budget=5,
    )

Usage (manual):
    from biosignals.agent import build_agent, get_feedback_store

    agent = build_agent(model_id="anthropic/claude-sonnet-4-20250514")
    result = agent.run("Run baseline experiment=galaxyppg_hr_ppg and report results.")
    store = get_feedback_store()
    print(store.history_for_agent())
"""
# from biosignals.agent.orchestrator import build_agent, run_experiment_loop
# from biosignals.agent.tools import get_feedback_store, reset_feedback_store
# from biosignals.agent.feedback import FeedbackStore, parse_run_dir, RunRecord, DriftReport

from .orchestrator import build_agent, run_experiment_loop
from .tools import get_feedback_store, reset_feedback_store
from .feedback import FeedbackStore, parse_run_dir, RunRecord, DriftReport

__all__ = [
    "build_agent",
    "run_experiment_loop",
    "get_feedback_store",
    "reset_feedback_store",
    "FeedbackStore",
    "parse_run_dir",
    "RunRecord",
    "DriftReport",
]