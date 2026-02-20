# src/biosignals/agent/dashboard.py
"""
Campaign dashboard generator.

Exports FeedbackStore data to JSON and generates a standalone HTML
dashboard that visualizes the experiment campaign:
  - Metrics progression across runs
  - Drift/stagnation detection events
  - Human approval decisions
  - Run comparison table
  - Agent recommendations

Usage:
    from biosignals.agent.dashboard import generate_dashboard

    # After a campaign completes:
    generate_dashboard(
        feedback_store=store,
        approval_hook=hook,       # optional
        output_dir="outputs/",
        campaign_goal="Minimize val/mae on GalaxyPPG",
        open_browser=True,
    )

    # Or from CLI:
    python -m biosignals.agent.dashboard --campaign-json campaign_data.json
"""

from __future__ import annotations

import json
import logging
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from biosignals.agent.feedback import FeedbackStore
    from biosignals.agent.hooks import ApprovalHook

log = logging.getLogger("biosignals.agent")

# ─────────────────────────────────────────────────
# Data export
# ─────────────────────────────────────────────────


def export_campaign_data(
    feedback_store: FeedbackStore,
    *,
    approval_hook: Optional[ApprovalHook] = None,
    campaign_goal: str = "",
    agent_summary: str = "",
) -> Dict[str, Any]:
    """
    Export all campaign data as a JSON-serializable dict.

    This is the bridge between the Python agent system and
    the HTML dashboard.
    """
    store_data = feedback_store.to_dict()
    drift = feedback_store.detect_drift()

    # Build per-run detail with training curves
    run_details = []
    for run in feedback_store.runs:
        curve = run.training_curve()
        detail = {
            "run_dir": run.run_dir,
            "task": run.task_name,
            "model": run.model_name,
            "dataset": run.dataset_name,
            "lr": run.lr,
            "epochs_configured": run.epochs_configured,
            "epochs_completed": run.n_epochs_completed,
            "best_value": run.best_monitor_value,
            "best_epoch": run.best_epoch,
            "final_value": run.final_monitor_value,
            "monitor_metric": run.monitor_metric,
            "monitor_mode": run.monitor_mode,
            "converged": run.converged,
            "failed": run.failed,
            "overrides": run.overrides,
            "timestamp": run.timestamp,
            "training_curve": [{"epoch": e, "value": round(v, 6)} for e, v in curve],
        }

        # Add final train/val metrics if available
        if run.epoch_history:
            last = run.epoch_history[-1]
            detail["final_train_metrics"] = {k: round(v, 6) for k, v in last.train.items()}
            detail["final_val_metrics"] = {k: round(v, 6) for k, v in (last.val or {}).items()}

        run_details.append(detail)

    # Approval history
    approval_history = []
    if approval_hook is not None:
        approval_history = approval_hook.to_dict()

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "goal": campaign_goal,
            "agent_summary": agent_summary,
            "n_runs": feedback_store.n_runs,
        },
        "drift": {
            "detected": drift.drift_detected,
            "stagnation": drift.stagnation_detected,
            "recommendation": drift.recommendation,
            "trend_slope": drift.trend_slope,
            "reasons": drift.reasons,
            "best_value": drift.best_value,
            "best_run_dir": drift.best_run_dir,
        },
        "runs": run_details,
        "approvals": approval_history,
    }


# ─────────────────────────────────────────────────
# HTML dashboard generation
# ─────────────────────────────────────────────────


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Campaign Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap');

  :root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --border: #1e293b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --gradient-blue: linear-gradient(135deg, #3b82f6, #8b5cf6);
    --gradient-green: linear-gradient(135deg, #10b981, #06b6d4);
    --gradient-red: linear-gradient(135deg, #ef4444, #f59e0b);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Outfit', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
  }

  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Header */
  .header {
    padding: 2rem 2rem 1.5rem;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #0f1729 0%, var(--bg-primary) 100%);
  }
  .header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
  }
  .header .subtitle {
    color: var(--text-secondary);
    font-size: 0.875rem;
  }
  .header .goal {
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.875rem;
    color: var(--text-secondary);
  }

  /* Status cards row */
  .status-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    padding: 1.5rem 2rem;
  }
  .status-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s;
  }
  .status-card:hover { border-color: var(--accent-blue); }
  .status-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
  }
  .status-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }
  .status-card .detail {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }
  .value.good { color: var(--accent-green); }
  .value.bad { color: var(--accent-red); }
  .value.warn { color: var(--accent-amber); }
  .value.neutral { color: var(--accent-blue); }

  /* Chart containers */
  .chart-section {
    padding: 1.5rem 2rem;
  }
  .chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }
  .chart-card h2 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .chart-card h2 .icon { font-size: 1.1rem; }
  .chart-wrapper {
    position: relative;
    height: 300px;
  }

  /* Run table */
  .run-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  .run-table th {
    text-align: left;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .run-table td {
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid rgba(30,41,59,0.5);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
  }
  .run-table tr:hover td { background: var(--bg-card-hover); }
  .run-table .best-row td { background: rgba(16,185,129,0.08); }

  .badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .badge.ok { background: rgba(59,130,246,0.15); color: var(--accent-blue); }
  .badge.conv { background: rgba(16,185,129,0.15); color: var(--accent-green); }
  .badge.fail { background: rgba(239,68,68,0.15); color: var(--accent-red); }

  /* Approval log */
  .approval-entry {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(30,41,59,0.5);
    font-size: 0.8rem;
  }
  .approval-entry:last-child { border-bottom: none; }
  .approval-marker {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 0.1rem;
  }
  .approval-marker.approve { background: rgba(16,185,129,0.2); color: var(--accent-green); }
  .approval-marker.reject { background: rgba(239,68,68,0.2); color: var(--accent-red); }
  .approval-marker.modify { background: rgba(245,158,11,0.2); color: var(--accent-amber); }

  /* Agent summary */
  .summary-block {
    white-space: pre-wrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
    line-height: 1.7;
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
  }

  .no-data {
    text-align: center;
    padding: 3rem;
    color: var(--text-muted);
    font-size: 0.875rem;
  }

  /* Two column layout for charts */
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }
  @media (max-width: 900px) {
    .chart-grid { grid-template-columns: 1fr; }
    .status-row { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>

<div class="header">
  <h1>&#x1F9EA; Experiment Campaign Dashboard</h1>
  <div class="subtitle" id="subtitle">Loading...</div>
  <div class="goal" id="goal"></div>
</div>

<div class="status-row" id="statusRow"></div>

<div class="chart-section">
  <div class="chart-grid">
    <div class="chart-card">
      <h2><span class="icon">&#x1F4C8;</span> Metric Progression</h2>
      <div class="chart-wrapper"><canvas id="metricChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h2><span class="icon">&#x1F3AF;</span> Training Curves</h2>
      <div class="chart-wrapper"><canvas id="curveChart"></canvas></div>
    </div>
  </div>

  <div class="chart-card">
    <h2><span class="icon">&#x1F4CB;</span> Experiment Log</h2>
    <div style="overflow-x: auto;">
      <table class="run-table" id="runTable">
        <thead><tr>
          <th>#</th><th>Status</th><th>Model</th><th>LR</th>
          <th>Epochs</th><th>Best Value</th><th>Overrides</th>
        </tr></thead>
        <tbody id="runTableBody"></tbody>
      </table>
    </div>
  </div>

  <div class="chart-card" id="approvalSection" style="display:none;">
    <h2><span class="icon">&#x1F9D1;</span> Human-in-the-Loop Approvals</h2>
    <div id="approvalLog"></div>
  </div>

  <div class="chart-card" id="summarySection" style="display:none;">
    <h2><span class="icon">&#x1F916;</span> Agent Summary</h2>
    <div class="summary-block" id="agentSummary"></div>
  </div>
</div>

<script>
// ── Campaign data injected by Python ──
const CAMPAIGN_DATA = __CAMPAIGN_DATA_PLACEHOLDER__;

function init() {
  const data = CAMPAIGN_DATA;
  if (!data || !data.runs) {
    document.getElementById('subtitle').textContent = 'No data loaded';
    return;
  }

  // Header
  const n = data.runs.length;
  const ts = data.meta?.generated_at ? new Date(data.meta.generated_at).toLocaleString() : '';
  document.getElementById('subtitle').textContent = `${n} experiments · Generated ${ts}`;
  document.getElementById('goal').textContent = data.meta?.goal || 'No goal specified';

  // Status cards
  renderStatusCards(data);

  // Charts
  renderMetricChart(data);
  renderCurveChart(data);

  // Run table
  renderRunTable(data);

  // Approvals
  if (data.approvals && data.approvals.length > 0) {
    document.getElementById('approvalSection').style.display = 'block';
    renderApprovals(data);
  }

  // Agent summary
  if (data.meta?.agent_summary) {
    document.getElementById('summarySection').style.display = 'block';
    document.getElementById('agentSummary').textContent = data.meta.agent_summary;
  }
}

function renderStatusCards(data) {
  const runs = data.runs.filter(r => !r.failed);
  const drift = data.drift || {};
  const bestVal = drift.best_value;
  const mode = runs.length > 0 ? runs[0].monitor_mode : 'min';
  const metric = runs.length > 0 ? runs[0].monitor_metric : '?';

  // Improvement from first to best
  let improvement = null;
  if (runs.length >= 2) {
    const first = runs[0].best_value;
    const best = bestVal;
    if (first && best && first !== 0) {
      improvement = ((first - best) / Math.abs(first) * 100);
      if (mode === 'max') improvement = -improvement;
    }
  }

  const cards = [
    {
      label: 'Total Runs',
      value: data.runs.length,
      cls: 'neutral',
      detail: `${runs.length} successful, ${data.runs.length - runs.length} failed`
    },
    {
      label: `Best ${metric}`,
      value: bestVal != null ? bestVal.toFixed(4) : 'N/A',
      cls: 'good',
      detail: mode === 'min' ? 'lower is better' : 'higher is better'
    },
    {
      label: 'Improvement',
      value: improvement != null ? `${improvement.toFixed(1)}%` : 'N/A',
      cls: improvement != null && improvement > 0 ? 'good' : 'warn',
      detail: 'from first run to best'
    },
    {
      label: 'Drift',
      value: drift.detected ? 'Detected' : 'None',
      cls: drift.detected ? 'bad' : 'good',
      detail: drift.recommendation ? `rec: ${drift.recommendation}` : ''
    },
    {
      label: 'Stagnation',
      value: drift.stagnation ? 'Detected' : 'None',
      cls: drift.stagnation ? 'warn' : 'good',
      detail: drift.trend_slope != null ? `slope: ${drift.trend_slope.toFixed(4)}` : ''
    },
  ];

  const html = cards.map(c => `
    <div class="status-card">
      <div class="label">${c.label}</div>
      <div class="value ${c.cls}">${c.value}</div>
      <div class="detail">${c.detail}</div>
    </div>
  `).join('');
  document.getElementById('statusRow').innerHTML = html;
}

function renderMetricChart(data) {
  const runs = data.runs.filter(r => !r.failed);
  if (runs.length === 0) return;

  const ctx = document.getElementById('metricChart').getContext('2d');
  const labels = runs.map((_, i) => `Run ${i + 1}`);
  const bestValues = runs.map(r => r.best_value);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: runs[0].monitor_metric + ' (best)',
        data: bestValues,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        borderWidth: 2,
        pointRadius: 6,
        pointBackgroundColor: runs.map((r, i) => {
          const isBest = r.best_value === Math.min(...bestValues.filter(v => v != null));
          return isBest ? '#10b981' : '#3b82f6';
        }),
        pointBorderColor: '#0a0e17',
        pointBorderWidth: 2,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } },
        tooltip: {
          callbacks: {
            afterLabel: (ctx) => {
              const r = runs[ctx.dataIndex];
              return `Model: ${r.model}\nLR: ${r.lr}\nEpochs: ${r.epochs_completed}/${r.epochs_configured}`;
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(30,41,59,0.5)' } },
        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(30,41,59,0.5)' } },
      }
    }
  });
}

function renderCurveChart(data) {
  const runs = data.runs.filter(r => r.training_curve && r.training_curve.length > 0);
  if (runs.length === 0) return;

  const ctx = document.getElementById('curveChart').getContext('2d');
  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#ec4899','#84cc16'];

  const datasets = runs.map((r, i) => ({
    label: `${r.model} lr=${r.lr}`,
    data: r.training_curve.map(p => ({ x: p.epoch, y: p.value })),
    borderColor: colors[i % colors.length],
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  }));

  new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } } },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Epoch', color: '#64748b' },
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(30,41,59,0.5)' },
        },
        y: {
          title: { display: true, text: runs[0].monitor_metric, color: '#64748b' },
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(30,41,59,0.5)' },
        },
      }
    }
  });
}

function renderRunTable(data) {
  const runs = data.runs;
  if (runs.length === 0) {
    document.getElementById('runTableBody').innerHTML = '<tr><td colspan="7" class="no-data">No runs</td></tr>';
    return;
  }

  const mode = runs[0].monitor_mode;
  const validValues = runs.filter(r => !r.failed).map(r => r.best_value);
  const bestVal = mode === 'min' ? Math.min(...validValues) : Math.max(...validValues);

  const rows = runs.map((r, i) => {
    const isBest = !r.failed && r.best_value === bestVal;
    const status = r.failed ? 'fail' : (r.converged ? 'conv' : 'ok');
    const statusLabel = r.failed ? 'FAIL' : (r.converged ? 'CONV' : 'OK');
    const overrides = (r.overrides || []).join(' ') || '(default)';

    return `<tr class="${isBest ? 'best-row' : ''}">
      <td>${i + 1}${isBest ? ' ★' : ''}</td>
      <td><span class="badge ${status}">${statusLabel}</span></td>
      <td>${r.model}</td>
      <td>${r.lr}</td>
      <td>${r.epochs_completed}/${r.epochs_configured}</td>
      <td>${r.failed ? '—' : r.best_value?.toFixed(4)}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${overrides}">${overrides}</td>
    </tr>`;
  }).join('');

  document.getElementById('runTableBody').innerHTML = rows;
}

function renderApprovals(data) {
  const approvals = data.approvals || [];
  if (approvals.length === 0) return;

  const html = approvals.map(a => {
    const icon = a.action === 'approve' ? '✓' : a.action === 'reject' ? '✕' : '✎';
    const cls = a.action === 'reject' ? 'reject' : a.action === 'modify' ? 'modify' : 'approve';
    return `
      <div class="approval-entry">
        <div class="approval-marker ${cls}">${icon}</div>
        <div>
          <div><strong>Run #${a.run_number}</strong> — ${a.action}</div>
          <div style="color:var(--text-muted);font-size:0.75rem;">
            Proposed: <span class="mono">${a.proposed}</span>
          </div>
          ${a.final !== a.proposed ? `<div style="color:var(--accent-amber);font-size:0.75rem;">Final: <span class="mono">${a.final}</span></div>` : ''}
          ${a.reason ? `<div style="color:var(--text-muted);font-size:0.7rem;margin-top:0.2rem;">${a.reason}</div>` : ''}
        </div>
      </div>
    `;
  }).join('');

  document.getElementById('approvalLog').innerHTML = html;
}

init();
</script>
</body>
</html>"""


def generate_dashboard(
    feedback_store: FeedbackStore,
    *,
    approval_hook: Optional[ApprovalHook] = None,
    output_dir: str | Path = ".",
    campaign_goal: str = "",
    agent_summary: str = "",
    open_browser: bool = True,
    filename: str = "campaign_dashboard.html",
) -> Path:
    """
    Generate a standalone HTML dashboard from campaign data.

    Args:
        feedback_store: The FeedbackStore with all experiment data.
        approval_hook: Optional ApprovalHook with human decision history.
        output_dir: Where to write the HTML file.
        campaign_goal: The campaign's goal string.
        agent_summary: The agent's final text summary.
        open_browser: Whether to auto-open in the default browser.
        filename: Output filename.

    Returns:
        Path to the generated HTML file.
    """
    # Export data
    data = export_campaign_data(
        feedback_store,
        approval_hook=approval_hook,
        campaign_goal=campaign_goal,
        agent_summary=agent_summary,
    )

    # Also write raw JSON (useful for programmatic access)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "campaign_data.json"
    json_path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Wrote campaign data: %s", json_path)

    # Generate HTML with embedded data
    data_json = json.dumps(data, default=str)
    html = _DASHBOARD_HTML.replace(
        "__CAMPAIGN_DATA_PLACEHOLDER__",
        data_json,
    )

    html_path = out_dir / filename
    html_path.write_text(html, encoding="utf-8")
    log.info("Wrote dashboard: %s", html_path)

    if open_browser:
        try:
            webbrowser.open(f"file://{html_path.resolve()}")
            log.info("Opened dashboard in browser")
        except Exception as e:
            log.warning("Could not open browser: %s", e)
            print(f"\n  Dashboard ready: file://{html_path.resolve()}\n")

    return html_path


# ─────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────


def main() -> None:
    """Generate dashboard from a campaign_data.json file."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate campaign dashboard")
    parser.add_argument(
        "--campaign-json",
        type=str,
        default="campaign_data.json",
        help="Path to campaign_data.json",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    # Load existing JSON data
    json_path = Path(args.campaign_json)
    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Generate HTML directly from JSON
    data_json = json.dumps(data, default=str)
    html = _DASHBOARD_HTML.replace("__CAMPAIGN_DATA_PLACEHOLDER__", data_json)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "campaign_dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"Dashboard: file://{html_path.resolve()}")
    if not args.no_open:
        webbrowser.open(f"file://{html_path.resolve()}")


if __name__ == "__main__":
    main()
