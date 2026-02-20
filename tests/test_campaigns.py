# tests/test_campaigns.py
"""
Unit tests for campaign goal management.
No API key needed.

Run:
    PYTHONPATH=$PWD/src python -m pytest tests/test_campaigns.py -v
"""
import textwrap
from pathlib import Path

import pytest

from biosignals.agent.campaigns import (
    CampaignGoal,
    load_goal,
    load_goal_from_inline,
    list_campaigns,
    _load_campaign_file,
)


# ─────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────


@pytest.fixture
def campaigns_dir(tmp_path):
    """Create a temporary campaigns directory with test data."""
    camp_dir = tmp_path / "campaigns"
    camp_dir.mkdir()

    # Write galaxyppg campaign
    (camp_dir / "galaxyppg.yaml").write_text(textwrap.dedent("""\
        dataset: galaxyppg
        description: "Galaxy PPG heart rate dataset"

        goals:
          hr_baseline:
            description: "Quick baseline"
            goal: |
              Minimize val/mae for HR regression on GalaxyPPG.
              Start with experiment=galaxyppg_hr_ppg.
              Budget: 3 runs. Target: val/mae < 15.
            budget: 3
            max_steps: 30
            baseline_experiment: galaxyppg_hr_ppg
            target_metric: val/mae
            target_value: 15.0
            target_direction: min
            tags: [baseline, quick]

          hr_deep:
            description: "Deep exploration"
            goal: |
              Minimize val/mae with architecture search.
            budget: 8
            max_steps: 60
            baseline_experiment: galaxyppg_hr_ppg
            target_metric: val/mae
            target_value: 5.0
            target_direction: min
            tags: [deep, tier3]
    """))

    # Write mitbih campaign
    (camp_dir / "mitbih.yaml").write_text(textwrap.dedent("""\
        dataset: mitbih
        description: "MIT-BIH arrhythmia database"

        goals:
          aami3_baseline:
            description: "AAMI-3 classification baseline"
            goal: |
              Maximize val/acc for AAMI-3 classification on MIT-BIH.
              Start with experiment=mitbih_aami3_cnn.
              Budget: 3 runs. Target: val/acc > 0.85.
            budget: 3
            max_steps: 30
            baseline_experiment: mitbih_aami3_cnn
            target_metric: val/acc
            target_value: 0.85
            target_direction: max
            tags: [baseline, classification]

          aami3_optimize:
            description: "Systematic optimization"
            goal: |
              Maximize val/acc for AAMI-3 with HP search.
            budget: 5
            max_steps: 45
            baseline_experiment: mitbih_aami3_cnn
            target_metric: val/acc
            target_value: 0.90
            target_direction: max
            tags: [optimization, classification]
    """))

    return camp_dir


# ═════════════════════════════════════════════════
# Tests: CampaignGoal
# ═════════════════════════════════════════════════


class TestCampaignGoal:
    def test_to_run_kwargs(self):
        goal = CampaignGoal(
            dataset="test", goal_name="g1", ref="test:g1",
            goal="Do something", budget=3, max_steps=30,
        )
        kwargs = goal.to_run_kwargs()
        assert kwargs["goal"] == "Do something"
        assert kwargs["budget"] == 3
        assert kwargs["max_steps"] == 30

    def test_summary(self):
        goal = CampaignGoal(
            dataset="mitbih", goal_name="aami3", ref="mitbih:aami3",
            goal="Maximize val/acc", budget=5, max_steps=45,
            description="Classification baseline",
            target_metric="val/acc", target_value=0.85,
            target_direction="max", tags=["baseline"],
        )
        s = goal.summary()
        assert "mitbih:aami3" in s
        assert "Classification baseline" in s
        assert "val/acc" in s
        assert "0.85" in s

    def test_inline_goal(self):
        goal = load_goal_from_inline("Minimize loss", budget=3, max_steps=20)
        assert goal.dataset == "inline"
        assert goal.goal == "Minimize loss"
        assert goal.budget == 3


# ═════════════════════════════════════════════════
# Tests: list_campaigns
# ═════════════════════════════════════════════════


class TestListCampaigns:
    def test_lists_all_goals(self, campaigns_dir):
        campaigns = list_campaigns(campaigns_dir=str(campaigns_dir))
        assert len(campaigns) == 4  # 2 galaxyppg + 2 mitbih

    def test_refs_are_correct(self, campaigns_dir):
        campaigns = list_campaigns(campaigns_dir=str(campaigns_dir))
        refs = {c["ref"] for c in campaigns}
        assert refs == {
            "galaxyppg:hr_baseline",
            "galaxyppg:hr_deep",
            "mitbih:aami3_baseline",
            "mitbih:aami3_optimize",
        }

    def test_includes_metadata(self, campaigns_dir):
        campaigns = list_campaigns(campaigns_dir=str(campaigns_dir))
        baseline = next(c for c in campaigns if c["ref"] == "mitbih:aami3_baseline")
        assert baseline["budget"] == 3
        assert baseline["target_metric"] == "val/acc"
        assert baseline["target_direction"] == "max"
        assert "classification" in baseline["tags"]

    def test_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        campaigns = list_campaigns(campaigns_dir=str(empty))
        assert campaigns == []


# ═════════════════════════════════════════════════
# Tests: load_goal
# ═════════════════════════════════════════════════


class TestLoadGoal:
    def test_load_galaxyppg_baseline(self, campaigns_dir):
        goal = load_goal("galaxyppg:hr_baseline", campaigns_dir=str(campaigns_dir))
        assert goal.dataset == "galaxyppg"
        assert goal.goal_name == "hr_baseline"
        assert goal.ref == "galaxyppg:hr_baseline"
        assert "val/mae" in goal.goal
        assert goal.budget == 3
        assert goal.target_metric == "val/mae"
        assert goal.target_direction == "min"
        assert goal.baseline_experiment == "galaxyppg_hr_ppg"

    def test_load_mitbih_baseline(self, campaigns_dir):
        goal = load_goal("mitbih:aami3_baseline", campaigns_dir=str(campaigns_dir))
        assert goal.dataset == "mitbih"
        assert goal.target_direction == "max"
        assert goal.target_value == 0.85
        assert goal.baseline_experiment == "mitbih_aami3_cnn"

    def test_budget_override(self, campaigns_dir):
        goal = load_goal(
            "galaxyppg:hr_baseline",
            campaigns_dir=str(campaigns_dir),
            budget_override=10,
        )
        assert goal.budget == 10  # overridden, not 3

    def test_max_steps_override(self, campaigns_dir):
        goal = load_goal(
            "galaxyppg:hr_deep",
            campaigns_dir=str(campaigns_dir),
            max_steps_override=100,
        )
        assert goal.max_steps == 100  # overridden, not 60

    def test_invalid_ref_format(self, campaigns_dir):
        with pytest.raises(ValueError, match="Invalid campaign ref"):
            load_goal("just_a_string", campaigns_dir=str(campaigns_dir))

    def test_missing_dataset(self, campaigns_dir):
        with pytest.raises(FileNotFoundError, match="No campaign file"):
            load_goal("nonexistent:goal", campaigns_dir=str(campaigns_dir))

    def test_missing_goal(self, campaigns_dir):
        with pytest.raises(KeyError, match="not found"):
            load_goal("galaxyppg:nonexistent_goal", campaigns_dir=str(campaigns_dir))


# ═════════════════════════════════════════════════
# Tests: CLI (run.py)
# ═════════════════════════════════════════════════


class TestRunCLI:
    def test_list_mode(self, campaigns_dir, capsys):
        """--list should print available campaigns and exit."""
        from biosignals.agent.run import main

        main(["--list", "--campaigns-dir", str(campaigns_dir)])
        captured = capsys.readouterr()
        assert "galaxyppg:hr_baseline" in captured.out
        assert "mitbih:aami3_baseline" in captured.out

    def test_no_args_errors(self):
        """No arguments should produce an error."""
        from biosignals.agent.run import main

        with pytest.raises(SystemExit):
            main([])


# ═════════════════════════════════════════════════
# Tests: Validation
# ═════════════════════════════════════════════════


class TestCampaignValidation:
    def test_missing_dataset_field(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("goals:\n  g1:\n    goal: test\n")
        with pytest.raises(ValueError, match="missing required 'dataset'"):
            _load_campaign_file(bad)

    def test_missing_goals_field(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("dataset: test\n")
        with pytest.raises(ValueError, match="missing 'goals'"):
            _load_campaign_file(bad)

    def test_empty_goal_text(self, campaigns_dir):
        # Add a goal with empty text
        p = campaigns_dir / "broken.yaml"
        p.write_text(
            "dataset: broken\ngoals:\n  empty:\n    goal: ''\n    budget: 1\n"
        )
        with pytest.raises(ValueError, match="empty goal text"):
            load_goal("broken:empty", campaigns_dir=str(campaigns_dir))