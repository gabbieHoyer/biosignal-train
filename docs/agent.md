# Feedback-Driven Experiment Agent

An autonomous experiment orchestration system that runs ML training experiments, analyzes results, detects performance drift, and iteratively improves model configurations — with optional human-in-the-loop approval at each step.

Built on [smolagents](https://github.com/huggingface/smolagents) (code-first agent framework) with a tiered capability system that escalates from hyperparameter tuning to novel architecture generation based on feedback signals.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Campaign Goal                        │
│              (campaigns/*.yaml or inline)               │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              smolagents CodeAgent                       │
│         (orchestrator.py + SYSTEM_PROMPT)               │
│                                                         │
│   Writes Python code at each step to call tools.        │
│   Reasons about results. Decides what to try next.      │
│                                                         │
│   ┌───────────┐  ┌───────────┐  ┌──────────────────┐    │
│   │  Tier 1   │  │  Tier 2   │  │     Tier 3       │    │
│   │ Read-only │  │ Write     │  │  Write Python    │    │
│   │ discovery │  │ YAML      │  │  (AST-validated) │    │
│   └───────────┘  └───────────┘  └──────────────────┘    │
└───────────────────────────┬─────────────────────────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
      ┌─────────────┐ ┌────────────┐ ┌──────────┐
      │ run_training│ │ Approval   │ │ Feedback │
      │ (subprocess)│ │ Hook       │ │ Store    │
      │             │ │ (optional) │ │          │
      │ python -m   │ │ approve/   │ │ drift    │
      │ biosignals  │ │ modify/    │ │ detect   │
      │ .cli.train  │ │ reject     │ │ stagnate │
      └──────┬──────┘ └────────────┘ └──────────┘
             │
             ▼
      ┌──────────────┐
      │ Run Artifacts│
      │ metrics.jsonl│
      │ summary.json │
      │ config.yaml  │
      └──────────────┘
```

**Two-loop design:**
- **Inner loop** (seconds/minutes): Agent reasons about which tool to call, executes code, reads results
- **Outer loop** (hours/days): FeedbackStore accumulates runs across a campaign, detects drift/stagnation, guides escalation decisions

## Capability Tiers

The agent has three capability tiers that form an escalation ladder — it starts with cheap interventions and only escalates to expensive ones when feedback signals warrant it.

### Tier 1 — Config-Aware (read-only)

The agent discovers what exists and analyzes results. Always active.

| Tool | Purpose |
|------|---------|
| `list_available_configs()` | Scan all config groups (models, experiments, datasets, trainers) |
| `suggest_search_space(run_dir)` | Data-driven next-step suggestions from completed run |
| `read_run_results(run_dir)` | Parse metrics, summary, config from run artifacts |
| `compare_runs(run_dirs)` | Side-by-side ranked comparison of multiple runs |
| `check_drift()` | Drift and stagnation analysis across campaign |
| `get_experiment_history()` | Compact history for agent context window |
| `read_experiment_config(run_dir)` | Resolved Hydra config from a run |

### Tier 2 — Config-Generating (write YAML)

The agent creates new Hydra configs from existing building blocks. Safe — YAML is data, not executable code. The `_target_` field (which determines what Python class is instantiated) is always preserved from the base config.

| Tool | Purpose |
|------|---------|
| `create_config_variant(group, base, name, mods)` | Clone a config and apply parameter changes |
| `compose_experiment_config(name, components, overrides)` | Compose a new experiment from component configs |

### Tier 3 — Code-Generating (write Python)

The agent writes new model architectures. All generated code is validated by an AST-based safety layer before being written to disk. Generated models live in an isolated directory (`src/biosignals/models/generated/`) and never touch human-authored code.

| Tool | Purpose |
|------|---------|
| `read_model_source(model_name)` | Read Python source of existing model architecture |
| `register_generated_model(name, code, args)` | AST-validate and register new nn.Module + Hydra config |

**Safety enforcement:**
- Import allowlist: `torch`, `numpy`, `scipy`, `einops`, `typing`, `math`, `biosignals`, safe stdlib
- Blocked calls: `exec`, `eval`, `open`, `input`, `getattr`, `setattr`, etc.
- Blocked attribute chains: `os.system`, `subprocess.run`, `shutil.rmtree`, `socket`, `http`
- Structural requirements: must define `nn.Module` subclass with `forward()` method
- No `global`/`nonlocal` statements

## Campaign Goals

Campaign goals are YAML files that define what the agent should optimize. They live in [`campaigns/`](../campaigns/) alongside `configs/`.

```bash
# List available goals
python -m biosignals.agent.run --list

# Run a goal
python -m biosignals.agent.run mitbih:aami3_baseline
python -m biosignals.agent.run galaxyppg:hr_optimize --budget 5
```

See [`campaigns/README.md`](../campaigns/README.md) for the full format specification.

## Human-in-the-Loop Approval

The approval hook intercepts before each `run_training()` call, showing the human:
- The agent's proposed experiment overrides
- Current campaign history (runs completed, best metric so far)
- Drift/stagnation status and recommendation

The human can approve, modify the overrides, reject (agent tries something else), or auto-approve the rest of the campaign.

```bash
# Interactive terminal mode
python -m biosignals.agent.run galaxyppg:hr_optimize --approval terminal
```

```
============================================================
  🔬 EXPERIMENT APPROVAL — Run #2
============================================================

  Proposed overrides:
    experiment=galaxyppg_hr_ppg trainer.lr=0.001 trainer.epochs=10

  History: 1 runs completed
  Current best: 50.7300 (val/mae min)
    Config: EncoderClassifier lr=0.0003
  Recommendation: continue

  Commands: [y]es  [n]o  [m] <new overrides>  [a]uto-approve-all  [?]help
------------------------------------------------------------
  →
```

The agent does not know the human is reviewing — the hook is transparent. This is intentional: the approval layer is orthogonal to the agent's reasoning, and can be added or removed without changing the agent's behavior.

**Programmatic hooks** are also available for automated policies:

```python
from biosignals.agent.hooks import CallbackApprovalHook, ApprovalDecision

# Reject any run with lr > 0.01
hook = CallbackApprovalHook(
    lambda overrides, store, run_num: (
        ApprovalDecision(action="reject", reason="lr too high")
        if "lr=0.1" in overrides
        else ApprovalDecision(action="approve")
    )
)
```

## Campaign Dashboard

After each campaign, a standalone HTML dashboard is generated with:
- **Status cards**: total runs, best metric, improvement percentage, drift/stagnation detection
- **Metric progression**: best value per run across the campaign
- **Training curves**: overlaid epoch-by-epoch curves for all runs
- **Experiment log**: ranked table with status badges, configs, metrics
- **Approval timeline**: shows human approve/reject/modify decisions
- **Agent summary**: the agent's final analysis text

The dashboard is a self-contained HTML file (Chart.js loaded from CDN). No server needed — open it in any browser, commit it to git, or attach it to a report.

```bash
# See a demo dashboard immediately (no API key, no training)
PYTHONPATH=$PWD/src python scripts/demo_dashboard.py
```

Example output: [`docs/examples/campaign_dashboard.html`](examples/campaign_dashboard.html)

## FeedbackStore and Drift Detection

The `FeedbackStore` is the core feedback component. It accumulates `RunRecord` objects across a campaign and provides actionable signals:

- **Drift detection**: Are recent runs *worse* than earlier runs? (degrading trend or regression from best)
- **Stagnation detection**: Are recent runs showing *no improvement* despite config changes?
- **Recommendations**: `"continue"`, `"change_lr"`, `"change_model"`, `"stop"`

These signals drive the agent's escalation decisions — when `check_drift()` returns `"change_model"` and all available models have been tried, the agent considers Tier 3 architecture generation.

## File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `tools.py` | ~1800 | All 12 smolagents @tool functions (Tier 1 + 2 + 3) |
| `feedback.py` | ~570 | RunRecord parsing, FeedbackStore, drift detection |
| `orchestrator.py` | ~320 | CodeAgent factory, system prompt, run_experiment_loop |
| `campaigns.py` | ~300 | Campaign goal loader (YAML → CampaignGoal) |
| `hooks.py` | ~300 | Human-in-the-loop approval hook system |
| `dashboard.py` | ~770 | Campaign dashboard HTML generator |
| `run.py` | ~220 | CLI entrypoint (`python -m biosignals.agent.run`) |
| `pipeline.py` | ~320 | Optional ZenML pipeline wrapper |

## Tests

All unit tests run without an API key:

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_feedback_store.py tests/test_config_discovery.py \
    tests/test_config_generation.py tests/test_code_generation.py tests/test_hooks.py \
    tests/test_dashboard.py tests/test_campaigns.py -v
```

Integration test (needs `ANTHROPIC_API_KEY`):

```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_simple_orchestration.py -v -m integration
```
