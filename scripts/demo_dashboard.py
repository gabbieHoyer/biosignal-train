#!/usr/bin/env python3
# scripts/demo_dashboard.py
"""
Generate a campaign dashboard from synthetic experiment data.

This creates realistic fake run data simulating a 5-run campaign
where the agent explores models and hyperparameters. The dashboard
opens in your browser — no training, no API key, no external services.

Usage (from project root):
    PYTHONPATH=$PWD/src python scripts/demo_dashboard.py

    # Or with custom output directory:
    PYTHONPATH=$PWD/src python scripts/demo_dashboard.py --output outputs/dashboard

    # Don't auto-open browser:
    PYTHONPATH=$PWD/src python scripts/demo_dashboard.py --no-open
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

# ─────────────────────────────────────────────────
# Ensure reproducible demo data
# ─────────────────────────────────────────────────
random.seed(42)


def make_epoch_records(
    n_epochs: int,
    start_val: float,
    end_val: float,
    noise: float = 0.03,
    monitor_metric: str = "val/mae",
    monitor_mode: str = "min",
):
    """Generate realistic epoch records with decaying metrics."""
    from biosignals.agent.feedback import EpochRecord

    records = []
    for e in range(n_epochs):
        t = e / max(n_epochs - 1, 1)
        # Exponential decay from start to end
        base = start_val + (end_val - start_val) * (1 - math.exp(-3 * t))
        val = max(0.01, base + random.uniform(-noise * base, noise * base))
        train_val = val * random.uniform(0.80, 0.95)  # train is better than val

        records.append(EpochRecord(
            epoch=e,
            global_step=e * 100,
            train={"loss": train_val * 0.5, "mae": train_val, "r2": max(0, 1 - train_val / 100)},
            val={"loss": val * 0.5, "mae": val, "r2": max(0, 1 - val / 100)},
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            monitor_value=val,
        ))
    return records


def build_demo_campaign():
    """
    Build a synthetic 5-run campaign that tells a realistic story:

    Run 1: Baseline (fast_dev, poor performance)
    Run 2: Increased epochs + tuned lr → big improvement
    Run 3: Different model (Transformer1D) → worse, agent learns
    Run 4: Back to best model, refined lr → improvement
    Run 5: Agent-generated deeper model → best result

    This mirrors the actual agent behavior we've seen in real campaigns.
    """
    from biosignals.agent.feedback import FeedbackStore, RunRecord
    from biosignals.agent.hooks import (
        AutoApproveHook,
        ApprovalDecision,
        CallbackApprovalHook,
    )

    store = FeedbackStore()

    # ── Run 1: Baseline ──
    epochs_1 = make_epoch_records(5, start_val=55.0, end_val=48.0)
    store.add_run(RunRecord(
        run_dir="/tmp/demo/run_2025-02-17_001",
        timestamp="2025-02-17T10:00:00",
        monitor_metric="val/mae", monitor_mode="min",
        best_epoch=4, best_value=48.23,
        best_ckpt_path="/tmp/demo/run_001/checkpoints/best.pt",
        last_epoch=4, last_ckpt_path="/tmp/demo/run_001/checkpoints/last.pt",
        epoch_history=epochs_1,
        task_name="RegressionTask", model_name="EncoderClassifier",
        dataset_name="GalaxyPPGDataset", lr=0.0003, epochs_configured=5,
        overrides=["experiment=galaxyppg_hr_ppg", "trainer=fast_dev"],
        config={},
    ))

    # ── Run 2: More epochs + higher lr → big improvement ──
    epochs_2 = make_epoch_records(10, start_val=42.0, end_val=10.5)
    store.add_run(RunRecord(
        run_dir="/tmp/demo/run_2025-02-17_002",
        timestamp="2025-02-17T10:15:00",
        monitor_metric="val/mae", monitor_mode="min",
        best_epoch=9, best_value=10.94,
        best_ckpt_path="/tmp/demo/run_002/checkpoints/best.pt",
        last_epoch=9, last_ckpt_path="/tmp/demo/run_002/checkpoints/last.pt",
        epoch_history=epochs_2,
        task_name="RegressionTask", model_name="EncoderClassifier",
        dataset_name="GalaxyPPGDataset", lr=0.001, epochs_configured=10,
        overrides=["experiment=galaxyppg_hr_ppg", "trainer.lr=0.001", "trainer.epochs=10"],
        config={},
    ))

    # ── Run 3: Try Transformer1D → worse (agent learns) ──
    epochs_3 = make_epoch_records(10, start_val=38.0, end_val=15.0)
    store.add_run(RunRecord(
        run_dir="/tmp/demo/run_2025-02-17_003",
        timestamp="2025-02-17T10:35:00",
        monitor_metric="val/mae", monitor_mode="min",
        best_epoch=8, best_value=15.22,
        best_ckpt_path="/tmp/demo/run_003/checkpoints/best.pt",
        last_epoch=9, last_ckpt_path="/tmp/demo/run_003/checkpoints/last.pt",
        epoch_history=epochs_3,
        task_name="RegressionTask", model_name="Transformer1D",
        dataset_name="GalaxyPPGDataset", lr=0.001, epochs_configured=10,
        overrides=["model=transformer1d", "trainer.lr=0.001", "trainer.epochs=10"],
        config={},
    ))

    # ── Run 4: Refined lr on best model → improvement ──
    epochs_4 = make_epoch_records(20, start_val=35.0, end_val=7.8)
    store.add_run(RunRecord(
        run_dir="/tmp/demo/run_2025-02-17_004",
        timestamp="2025-02-17T10:55:00",
        monitor_metric="val/mae", monitor_mode="min",
        best_epoch=17, best_value=8.31,
        best_ckpt_path="/tmp/demo/run_004/checkpoints/best.pt",
        last_epoch=19, last_ckpt_path="/tmp/demo/run_004/checkpoints/last.pt",
        epoch_history=epochs_4,
        task_name="RegressionTask", model_name="EncoderClassifier",
        dataset_name="GalaxyPPGDataset", lr=0.0005, epochs_configured=20,
        overrides=["experiment=galaxyppg_hr_ppg", "trainer.lr=0.0005", "trainer.epochs=20"],
        config={},
    ))

    # ── Run 5: Agent-generated deeper model → best ──
    epochs_5 = make_epoch_records(20, start_val=30.0, end_val=6.5)
    store.add_run(RunRecord(
        run_dir="/tmp/demo/run_2025-02-17_005",
        timestamp="2025-02-17T11:20:00",
        monitor_metric="val/mae", monitor_mode="min",
        best_epoch=18, best_value=7.12,
        best_ckpt_path="/tmp/demo/run_005/checkpoints/best.pt",
        last_epoch=19, last_ckpt_path="/tmp/demo/run_005/checkpoints/last.pt",
        epoch_history=epochs_5,
        task_name="RegressionTask", model_name="ResNet1DDeep",
        dataset_name="GalaxyPPGDataset", lr=0.0005, epochs_configured=20,
        overrides=["model=resnet1d_deep", "trainer.lr=0.0005", "trainer.epochs=20"],
        config={},
    ))

    # ── Approval hook history (simulate human-in-the-loop) ──
    hook = CallbackApprovalHook(
        lambda o, s, n: ApprovalDecision(action="approve", reason="looks good")
    )
    # Simulate: human approved runs 1-3, modified run 4, approved run 5
    hook._run_counter = 0

    hook.history = []
    from biosignals.agent.hooks import ApprovalRecord

    hook.history.append(ApprovalRecord(
        run_number=1, timestamp="2025-02-17T09:59:00",
        proposed_overrides="experiment=galaxyppg_hr_ppg trainer=fast_dev",
        decision=ApprovalDecision(action="approve", reason="baseline — let's see"),
        final_overrides="experiment=galaxyppg_hr_ppg trainer=fast_dev",
    ))
    hook.history.append(ApprovalRecord(
        run_number=2, timestamp="2025-02-17T10:14:00",
        proposed_overrides="experiment=galaxyppg_hr_ppg trainer.lr=0.001 trainer.epochs=10",
        decision=ApprovalDecision(action="approve", reason="good suggestion from agent"),
        final_overrides="experiment=galaxyppg_hr_ppg trainer.lr=0.001 trainer.epochs=10",
    ))
    hook.history.append(ApprovalRecord(
        run_number=3, timestamp="2025-02-17T10:34:00",
        proposed_overrides="model=transformer1d trainer.lr=0.001 trainer.epochs=10",
        decision=ApprovalDecision(action="approve", reason="worth trying different arch"),
        final_overrides="model=transformer1d trainer.lr=0.001 trainer.epochs=10",
    ))
    hook.history.append(ApprovalRecord(
        run_number=4, timestamp="2025-02-17T10:54:00",
        proposed_overrides="experiment=galaxyppg_hr_ppg trainer.lr=0.001 trainer.epochs=30",
        decision=ApprovalDecision(
            action="modify",
            modified_overrides="experiment=galaxyppg_hr_ppg trainer.lr=0.0005 trainer.epochs=20",
            reason="too many epochs, lower lr instead",
        ),
        final_overrides="experiment=galaxyppg_hr_ppg trainer.lr=0.0005 trainer.epochs=20",
    ))
    hook.history.append(ApprovalRecord(
        run_number=5, timestamp="2025-02-17T11:19:00",
        proposed_overrides="model=resnet1d_deep trainer.lr=0.0005 trainer.epochs=20",
        decision=ApprovalDecision(action="approve", reason="agent-generated arch, let's test it"),
        final_overrides="model=resnet1d_deep trainer.lr=0.0005 trainer.epochs=20",
    ))

    # ── Agent summary ──
    agent_summary = """Campaign Summary (5 runs):

Ranked results (best to worst):
  1. ResNet1DDeep  lr=0.0005  val/mae=7.12  (agent-generated architecture)
  2. EncoderClassifier  lr=0.0005  val/mae=8.31  (refined hyperparameters)
  3. EncoderClassifier  lr=0.001  val/mae=10.94  (first improvement from baseline)
  4. Transformer1D  lr=0.001  val/mae=15.22  (architecture exploration)
  5. EncoderClassifier  lr=0.0003  val/mae=48.23  (baseline, fast_dev)

Best configuration: model=resnet1d_deep trainer.lr=0.0005 trainer.epochs=20
  → val/mae = 7.12 (85% improvement over baseline)

Key learnings:
  - EncoderClassifier is a strong baseline when properly tuned
  - Transformer1D underperformed — likely needs more data or different hyperparameters
  - The agent-generated ResNet1DDeep (deeper residual blocks) achieved the best result
  - Learning rate 0.0005 was the sweet spot for this dataset
  - 20 epochs was sufficient — run 4 showed convergence around epoch 17

Next steps if budget were extended:
  - Try ResNet1DDeep with lr schedule (warmup + cosine decay)
  - Explore data augmentation via transforms configs
  - Create an ensemble of EncoderClassifier + ResNet1DDeep"""

    return store, hook, agent_summary


def main():
    parser = argparse.ArgumentParser(description="Generate demo campaign dashboard")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    print("Building synthetic campaign data...")
    store, hook, agent_summary = build_demo_campaign()

    print(f"  {store.n_runs} runs generated")
    print(f"  {len(hook.history)} approval records")
    print(f"  Best: {store.best_run().best_monitor_value:.2f} "
          f"({store.best_run().model_name})")

    print("\nGenerating dashboard...")
    from biosignals.agent.dashboard import generate_dashboard

    html_path = generate_dashboard(
        store,
        approval_hook=hook,
        output_dir=args.output,
        campaign_goal=(
            "Minimize val/mae for HR regression on GalaxyPPG. "
            "Start with experiment=galaxyppg_hr_ppg. "
            "Budget: 5 runs. Target: val/mae < 10."
        ),
        agent_summary=agent_summary,
        open_browser=not args.no_open,
    )

    print(f"\n  Dashboard: file://{html_path.resolve()}")
    print(f"  Raw data:  file://{(html_path.parent / 'campaign_data.json').resolve()}")
    print()
    if args.no_open:
        print("  Open the HTML file in any browser to view the dashboard.")


if __name__ == "__main__":
    main()
