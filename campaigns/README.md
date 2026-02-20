# Campaign Goals

Campaign files define **what the experiment agent should optimize** — separate from model configs (`configs/`) which define **how models are built**.

Each dataset gets its own YAML file with named goals at different levels of exploration depth.

## Quick start

```bash
# List all available campaigns
python -m biosignals.agent.run --list

# Run a named goal
python -m biosignals.agent.run mitbih:aami3_baseline
python -m biosignals.agent.run galaxyppg:hr_optimize --budget 5

# Run with interactive approval (human reviews each experiment)
python -m biosignals.agent.run mitbih:aami3_baseline --approval terminal

# Inline goal (no campaign file needed)
python -m biosignals.agent.run --goal "Minimize val/mae on GalaxyPPG. Budget: 3." --budget 3
```

## File format

```yaml
# campaigns/{dataset}.yaml

dataset: mitbih                          # matches configs/dataset/{name}.yaml
description: "MIT-BIH Arrhythmia Database"

goals:
  aami3_baseline:                        # selectable as mitbih:aami3_baseline
    description: "Fast baseline for AAMI-3 classification"
    goal: |                              # natural language prompt sent to the agent
      Maximize val/acc for AAMI-3 classification on MIT-BIH.
      Start with experiment=mitbih_aami3_cnn.
      Budget: 3 runs. Target: val/acc > 0.85.
    budget: 3                            # max training runs
    max_steps: 30                        # max agent reasoning steps
    baseline_experiment: mitbih_aami3_cnn
    target_metric: val/acc
    target_value: 0.85
    target_direction: max                # "min" for regression, "max" for classification
    tags: [baseline, classification]
```

## Adding a new dataset

1. Create `campaigns/{dataset}.yaml` following the format above
2. Verify with `python -m biosignals.agent.run --list`
3. Run with `python -m biosignals.agent.run {dataset}:{goal_name}`

The goal text should tell the agent:
- Which experiment config to start from
- What metric to optimize and in which direction
- Dataset-specific context (number of classes, modalities, known challenges like class imbalance)
- Budget and target

## Current campaigns

| File | Dataset | Goals |
|------|---------|-------|
| `galaxyppg.yaml` | GalaxyPPG PPG heart rate | `hr_baseline`, `hr_optimize`, `hr_deep` |
| `mitbih.yaml` | MIT-BIH AAMI-3 arrhythmia | `aami3_baseline`, `aami3_optimize`, `aami3_deep` |

See [`docs/agent.md`](../docs/agent.md) for full documentation of the agent system.