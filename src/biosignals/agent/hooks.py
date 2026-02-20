# src/biosignals/agent/hooks.py
"""
Human-in-the-loop approval hooks for experiment orchestration.

Provides a callback interface that intercepts before each training run,
allowing a human (or programmatic controller) to:
  - APPROVE the proposed experiment as-is
  - MODIFY the overrides before running
  - REJECT the experiment entirely
  - SKIP approval (auto-approve) for the rest of the campaign

This fills the gap that LangGraph's "interrupt" nodes provide,
but in a simpler, more flexible callback design.

Usage:
    from biosignals.agent.hooks import TerminalApprovalHook, set_approval_hook

    # Interactive terminal mode
    set_approval_hook(TerminalApprovalHook())

    # Or programmatic (for tests)
    set_approval_hook(AutoApproveHook())

    # Or custom
    set_approval_hook(lambda overrides, store, run_number: ApprovalDecision("approve"))
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from biosignals.agent.feedback import FeedbackStore

log = logging.getLogger("biosignals.agent")


# ─────────────────────────────────────────────────
# Decision dataclass
# ─────────────────────────────────────────────────


@dataclass
class ApprovalDecision:
    """Result of a human-in-the-loop approval check."""

    action: str  # "approve", "modify", "reject", "auto_approve_remaining"
    modified_overrides: Optional[str] = None  # only if action == "modify"
    reason: Optional[str] = None  # human's stated reason

    @property
    def approved(self) -> bool:
        return self.action in ("approve", "modify", "auto_approve_remaining")


@dataclass
class ApprovalRecord:
    """Logged record of each approval decision (for dashboard)."""

    run_number: int
    timestamp: str
    proposed_overrides: str
    decision: ApprovalDecision
    final_overrides: str  # what actually ran (may differ if modified)


# ─────────────────────────────────────────────────
# Hook interface
# ─────────────────────────────────────────────────


class ApprovalHook(ABC):
    """Base class for approval hooks."""

    def __init__(self) -> None:
        self.history: List[ApprovalRecord] = []
        self._auto_approve: bool = False
        self._run_counter: int = 0

    @abstractmethod
    def review(
        self,
        overrides: str,
        store: FeedbackStore,
        run_number: int,
    ) -> ApprovalDecision:
        """
        Present the proposed experiment to the reviewer.

        Args:
            overrides: The Hydra override string the agent wants to run.
            store: Current FeedbackStore with all experiment history.
            run_number: Which run this is (1-indexed).

        Returns:
            ApprovalDecision with the reviewer's choice.
        """
        ...

    def __call__(
        self,
        overrides: str,
        store: FeedbackStore,
    ) -> ApprovalDecision:
        """Called by run_training tool. Handles auto-approve and logging."""
        self._run_counter += 1

        if self._auto_approve:
            decision = ApprovalDecision(
                action="approve",
                reason="auto-approved (human chose to skip remaining)",
            )
        else:
            decision = self.review(overrides, store, self._run_counter)

        if decision.action == "auto_approve_remaining":
            self._auto_approve = True
            decision = ApprovalDecision(
                action="approve",
                reason="auto-approved (+ all remaining)",
            )

        final_overrides = decision.modified_overrides if decision.action == "modify" else overrides

        record = ApprovalRecord(
            run_number=self._run_counter,
            timestamp=datetime.now().isoformat(),
            proposed_overrides=overrides,
            decision=decision,
            final_overrides=final_overrides,
        )
        self.history.append(record)

        log.info(
            "Approval hook [run %d]: %s (overrides: %s)",
            self._run_counter,
            decision.action,
            final_overrides[:80],
        )
        return decision

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize approval history for dashboard."""
        return [
            {
                "run_number": r.run_number,
                "timestamp": r.timestamp,
                "proposed": r.proposed_overrides,
                "action": r.decision.action,
                "reason": r.decision.reason,
                "final": r.final_overrides,
            }
            for r in self.history
        ]


# ─────────────────────────────────────────────────
# Built-in hooks
# ─────────────────────────────────────────────────


class AutoApproveHook(ApprovalHook):
    """Auto-approves everything. Use for non-interactive / test runs."""

    def review(self, overrides, store, run_number):
        return ApprovalDecision(action="approve", reason="auto")


class TerminalApprovalHook(ApprovalHook):
    """
    Interactive terminal approval. Presents experiment details
    and waits for human input before each training run.

    Commands:
        y / yes / enter  → approve
        n / no           → reject
        m <overrides>    → modify overrides
        a / auto         → approve this + all remaining
        ? / help         → show commands
    """

    def review(self, overrides, store, run_number):
        # Build context display
        print("\n" + "=" * 60)
        print(f"  🔬 EXPERIMENT APPROVAL — Run #{run_number}")
        print("=" * 60)
        print("\n  Proposed overrides:")
        print(f"    {overrides}")

        # Show experiment history context
        if store.n_runs > 0:
            best = store.best_run()
            print(f"\n  History: {store.n_runs} runs completed")
            if best:
                print(
                    f"  Current best: {best.best_monitor_value:.4f} "
                    f"({best.monitor_metric} {best.monitor_mode})"
                )
                print(f"    Config: {best.model_name} lr={best.lr}")

            drift = store.detect_drift()
            if drift.drift_detected:
                print(f"  ⚠ DRIFT DETECTED: {drift.reasons[0] if drift.reasons else ''}")
            if drift.stagnation_detected:
                print(f"  ⚠ STAGNATION: {drift.reasons[0] if drift.reasons else ''}")
            print(f"  Recommendation: {drift.recommendation}")
        else:
            print("\n  (First run — no history yet)")

        print("\n  Commands: [y]es  [n]o  [m] <new overrides>  [a]uto-approve-all  [?]help")
        print("-" * 60)

        while True:
            try:
                response = input("  → ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  (interrupted — rejecting)")
                return ApprovalDecision(action="reject", reason="interrupted")

            if response in ("", "y", "yes"):
                return ApprovalDecision(action="approve", reason="human approved")

            elif response in ("n", "no"):
                reason = input("  Reason (optional): ").strip() or "human rejected"
                return ApprovalDecision(action="reject", reason=reason)

            elif response.startswith("m ") or response.startswith("m\t"):
                new_overrides = response[2:].strip()
                if not new_overrides:
                    print("  Usage: m experiment=X trainer.lr=0.001")
                    continue
                print(f"  Modified: {new_overrides}")
                return ApprovalDecision(
                    action="modify",
                    modified_overrides=new_overrides,
                    reason="human modified",
                )

            elif response in ("a", "auto"):
                return ApprovalDecision(
                    action="auto_approve_remaining",
                    reason="human chose auto-approve",
                )

            elif response in ("?", "help"):
                print("  y/yes/enter → approve this experiment")
                print("  n/no        → reject (agent will try something else)")
                print("  m <args>    → modify overrides (replace what agent proposed)")
                print("  a/auto      → approve this + auto-approve all remaining")
                print("  ?/help      → show this help")

            else:
                print(f"  Unknown command: '{response}'. Type ? for help.")


class CallbackApprovalHook(ApprovalHook):
    """
    Wraps a callable for programmatic approval.

    Usage:
        hook = CallbackApprovalHook(
            lambda overrides, store, run_number: ApprovalDecision("approve")
        )
    """

    def __init__(
        self,
        callback: Callable[[str, FeedbackStore, int], ApprovalDecision],
    ) -> None:
        super().__init__()
        self._callback = callback

    def review(self, overrides, store, run_number):
        return self._callback(overrides, store, run_number)


# ─────────────────────────────────────────────────
# Module-level hook (shared with tools.py)
# ─────────────────────────────────────────────────

_approval_hook: Optional[ApprovalHook] = None


def set_approval_hook(hook: Optional[ApprovalHook]) -> None:
    """Set the global approval hook. Pass None to disable."""
    global _approval_hook
    _approval_hook = hook
    if hook:
        log.info("Approval hook set: %s", type(hook).__name__)
    else:
        log.info("Approval hook disabled")


def get_approval_hook() -> Optional[ApprovalHook]:
    """Get the current approval hook (for tools.py to call)."""
    return _approval_hook


def clear_approval_hook() -> None:
    """Remove the approval hook."""
    set_approval_hook(None)
