# tests/test_hooks.py
"""
Unit tests for human-in-the-loop approval hooks.
No API key needed.

Run:
    PYTHONPATH=$PWD/src python -m pytest tests/test_hooks.py -v
"""

import json

import pytest

from biosignals.agent.feedback import FeedbackStore
from biosignals.agent.hooks import (
    ApprovalDecision,
    AutoApproveHook,
    CallbackApprovalHook,
    clear_approval_hook,
    get_approval_hook,
    set_approval_hook,
)

# ─────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────


@pytest.fixture
def store():
    return FeedbackStore()


@pytest.fixture(autouse=True)
def clean_global_hook():
    """Ensure global hook is cleared after each test."""
    yield
    clear_approval_hook()


# ─────────────────────────────────────────────────
# Tests: ApprovalDecision
# ─────────────────────────────────────────────────


class TestApprovalDecision:
    def test_approve(self):
        d = ApprovalDecision(action="approve")
        assert d.approved is True

    def test_reject(self):
        d = ApprovalDecision(action="reject", reason="bad idea")
        assert d.approved is False

    def test_modify(self):
        d = ApprovalDecision(action="modify", modified_overrides="trainer.lr=0.01")
        assert d.approved is True

    def test_auto_approve_remaining(self):
        d = ApprovalDecision(action="auto_approve_remaining")
        assert d.approved is True


# ─────────────────────────────────────────────────
# Tests: AutoApproveHook
# ─────────────────────────────────────────────────


class TestAutoApproveHook:
    def test_approves_everything(self, store):
        hook = AutoApproveHook()
        d1 = hook("experiment=foo trainer.lr=0.001", store)
        d2 = hook("experiment=bar", store)
        d3 = hook("model=transformer1d", store)

        assert d1.approved is True
        assert d2.approved is True
        assert d3.approved is True

    def test_tracks_history(self, store):
        hook = AutoApproveHook()
        hook("overrides1", store)
        hook("overrides2", store)

        assert len(hook.history) == 2
        assert hook.history[0].run_number == 1
        assert hook.history[1].run_number == 2
        assert hook.history[0].proposed_overrides == "overrides1"

    def test_serializes_to_dict(self, store):
        hook = AutoApproveHook()
        hook("overrides1", store)
        hook("overrides2", store)

        records = hook.to_dict()
        assert len(records) == 2
        assert records[0]["run_number"] == 1
        assert records[0]["action"] == "approve"
        assert "timestamp" in records[0]


# ─────────────────────────────────────────────────
# Tests: CallbackApprovalHook
# ─────────────────────────────────────────────────


class TestCallbackApprovalHook:
    def test_reject_on_condition(self, store):
        """Reject any run with lr > 0.01."""

        def policy(overrides, st, run_num):
            if "lr=0.1" in overrides:
                return ApprovalDecision(action="reject", reason="lr too high")
            return ApprovalDecision(action="approve")

        hook = CallbackApprovalHook(policy)

        d1 = hook("trainer.lr=0.001", store)
        d2 = hook("trainer.lr=0.1", store)

        assert d1.approved is True
        assert d2.approved is False
        assert d2.reason == "lr too high"

    def test_modify_overrides(self, store):
        """Always add epochs=5 to the overrides."""

        def policy(overrides, st, run_num):
            return ApprovalDecision(
                action="modify",
                modified_overrides=overrides + " trainer.epochs=5",
            )

        hook = CallbackApprovalHook(policy)
        d = hook("experiment=foo", store)

        assert d.approved is True
        assert d.action == "modify"
        assert hook.history[0].final_overrides == "experiment=foo trainer.epochs=5"

    def test_auto_approve_remaining(self, store):
        """First run asks, then auto-approve the rest."""
        call_count = 0

        def policy(overrides, st, run_num):
            nonlocal call_count
            call_count += 1
            if run_num == 1:
                return ApprovalDecision(action="auto_approve_remaining")
            return ApprovalDecision(action="approve")

        hook = CallbackApprovalHook(policy)

        # Run 1: triggers auto-approve
        d1 = hook("run1", store)
        assert d1.approved is True

        # Run 2+: should be auto-approved without calling policy
        d2 = hook("run2", store)
        d3 = hook("run3", store)
        assert d2.approved is True
        assert d3.approved is True
        # Policy should only have been called once (run 1)
        assert call_count == 1


# ─────────────────────────────────────────────────
# Tests: Global hook management
# ─────────────────────────────────────────────────


class TestGlobalHook:
    def test_set_and_get(self):
        hook = AutoApproveHook()
        set_approval_hook(hook)
        assert get_approval_hook() is hook

    def test_clear(self):
        set_approval_hook(AutoApproveHook())
        clear_approval_hook()
        assert get_approval_hook() is None

    def test_none_by_default(self):
        assert get_approval_hook() is None


# ─────────────────────────────────────────────────
# Tests: Hook integration with run_training
# ─────────────────────────────────────────────────


class TestRunTrainingHookIntegration:
    """Test that run_training respects the approval hook.

    These test the hook wiring only — they don't run real training.
    The hook fires before subprocess.run, so a rejection means
    subprocess.run is never called.
    """

    def test_rejection_prevents_training(self, store):
        """When hook rejects, run_training returns error without launching subprocess."""
        reject_hook = CallbackApprovalHook(
            lambda o, s, n: ApprovalDecision(action="reject", reason="test")
        )
        set_approval_hook(reject_hook)

        from biosignals.agent.tools import run_training

        result = json.loads(run_training.forward(overrides="experiment=test"))
        assert result["success"] is False
        assert result["returncode"] == -2
        assert "rejected" in result["error"]

    def test_modification_changes_overrides(self, store):
        """When hook modifies, the modified overrides are used.

        We can't verify the actual subprocess call easily, but we can
        verify the hook records the modification.
        """
        modify_hook = CallbackApprovalHook(
            lambda o, s, n: ApprovalDecision(
                action="modify",
                modified_overrides="trainer.lr=0.0001",
            )
        )
        set_approval_hook(modify_hook)

        # The training will fail (no real CLI), but the hook should fire first
        from biosignals.agent.tools import run_training

        run_training.forward(overrides="trainer.lr=0.1")

        assert len(modify_hook.history) == 1
        assert modify_hook.history[0].proposed_overrides == "trainer.lr=0.1"
        assert modify_hook.history[0].final_overrides == "trainer.lr=0.0001"

    def test_no_hook_means_no_interference(self, store):
        """When no hook is set, run_training proceeds normally."""
        clear_approval_hook()

        from biosignals.agent.tools import run_training

        # Will fail because there's no real training CLI, but it should
        # attempt to run (returncode != -2)
        result = json.loads(run_training.forward(overrides="experiment=test"))
        assert result["returncode"] != -2  # -2 means hook rejected
