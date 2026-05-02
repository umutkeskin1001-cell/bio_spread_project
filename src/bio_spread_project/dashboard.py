from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_dashboard(audit: dict[str, Any], output_path: Path) -> Path:
    """Generate a clear, standalone audit dashboard with client-side rendering."""
    payload = {
        "validation": audit.get("validation", {}),
        "quality_gates": audit.get("quality_gates", {}),
        "all_passed": bool(audit.get("all_quality_gates_passed", False)),
        "primary_model": audit.get("primary_model", "unknown"),
        "input_mode": audit.get("input_mode", "unknown"),
        "environment": audit.get("environment", {}),
        "input_hashes": audit.get("input_hashes", {}),
    }

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BioSpread Reliability Audit</title>
  <style>
    :root {{
      --bg: #f7f8fb; --paper: #ffffff; --ink: #18202f; --muted: #617089;
      --line: #d9e0ea; --blue: #2457d6; --green: #14845f; --red: #bf2f45;
      --amber: #a96700; --shadow: 0 10px 30px rgba(30, 41, 59, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: system-ui, -apple-system, sans-serif; line-height: 1.45; }}
    .shell {{ max-width: 1200px; margin: 0 auto; padding: 40px 24px; }}
    header {{ display: flex; justify-content: space-between; align-items: start; padding: 32px; background: var(--paper); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p {{ margin: 0; color: var(--muted); }}
    .status {{ padding: 12px 24px; border-radius: 8px; font-weight: 700; border: 1px solid; }}
    .status.pass {{ color: var(--green); background: #ecf8f3; border-color: #b8e4d2; }}
    .status.fail {{ color: var(--red); background: #fff5f5; border-color: #feb2b2; }}
    .grid {{ display: grid; gap: 24px; margin-top: 24px; }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
    .two-col {{ grid-template-columns: 1.2fr 0.8fr; }}
    .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px; padding: 24px; box-shadow: var(--shadow); }}
    .metric-label {{ color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-value {{ margin-top: 8px; font-size: 32px; font-weight: 800; }}
    .table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    .table th {{ text-align: left; color: var(--muted); padding: 12px 8px; border-bottom: 2px solid var(--line); }}
    .table td {{ padding: 12px 8px; border-bottom: 1px solid var(--line); }}
    .gate-pass {{ color: var(--green); font-weight: 600; }}
    .gate-fail {{ color: var(--red); font-weight: 600; }}
    svg {{ width: 100%; height: auto; }}
    @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>BioSpread Reliability Audit</h1>
        <p>What this predicts: geographic spread risk signals for plasmid backbones.</p>
      </div>
      <div id="main-status" class="status">LOADING...</div>
    </header>

    <div class="grid metrics" id="metric-grid"></div>

    <div class="grid two-col">
      <div class="card">
        <h3>Calibration Curve</h3>
        <div id="calibration-container"></div>
      </div>
      <div class="card">
        <h3>Quality Gate Detail</h3>
        <table class="table"><tbody id="gate-table"></tbody></table>
      </div>
    </div>

    <div class="grid two-col">
      <div class="card">
        <h3>Permutation Importance</h3>
        <div id="importance-container"></div>
      </div>
      <div class="card">
        <h3>Environment</h3>
        <table class="table" id="env-table"></table>
      </div>
    </div>
  </div>

  <script>
    const data = {json.dumps(payload)};

    // 1. Header Status
    const statusEl = document.getElementById('main-status');
    statusEl.textContent = data.all_passed ? 'QUALITY PASS' : 'REVIEW REQUIRED';
    statusEl.className = 'status ' + (data.all_passed ? 'pass' : 'fail');

    // 2. Metrics
    const metrics = [
      {{ label: 'ROC AUC', value: data.validation.roc_auc }},
      {{ label: 'Avg Precision', value: data.validation.average_precision }},
      {{ label: 'Calibration ECE', value: data.validation.expected_calibration_error }},
      {{ label: 'Prevalence', value: data.validation.prevalence }},
    ];
    const grid = document.getElementById('metric-grid');
    metrics.forEach(m => {{
      const card = document.createElement('div');
      card.className = 'card';
      const label = document.createElement('div');
      label.className = 'metric-label';
      label.textContent = m.label;
      const value = document.createElement('div');
      value.className = 'metric-value';
      value.textContent = Number.isFinite(m.value) ? m.value.toFixed(3) : 'not_evaluated';
      card.appendChild(label);
      card.appendChild(value);
      grid.appendChild(card);
    }});

    // 3. Gates
    const gateTable = document.getElementById('gate-table');
    Object.entries(data.quality_gates).forEach(([name, pass]) => {{
      const tr = document.createElement('tr');
      const tdName = document.createElement('td');
      tdName.textContent = name.replace(/_/g, ' ');
      const tdPass = document.createElement('td');
      tdPass.className = pass ? 'gate-pass' : 'gate-fail';
      tdPass.textContent = pass ? 'PASS' : 'FAIL';
      tr.appendChild(tdName);
      tr.appendChild(tdPass);
      gateTable.appendChild(tr);
    }});

    // 4. Calibration SVG
    const calBins = data.validation.calibration_bins || [];
    if (calBins.length > 0) {{
      const w = 600, h = 300, pad = 40;
      let svg = `<svg viewBox="0 0 ${{w}} ${{h}}">`;
      svg += `<line x1="${{pad}}" y1="${{h-pad}}" x2="${{w-pad}}" y2="${{pad}}" stroke="#cbd5e1" stroke-dasharray="4"/>`;

      const pts = calBins.map((b, i) => {{
        const x = pad + (i / (calBins.length - 1)) * (w - 2 * pad);
        const y_pred = h - pad - (b.mean_prediction * (h - 2 * pad));
        const y_obs = h - pad - (b.observed_rate * (h - 2 * pad));
        return {{x, y_pred, y_obs}};
      }});

      const line = (key, color) => {{
        const d = pts.map((p, i) => (i === 0 ? 'M' : 'L') + `${{p.x}},${{p[key]}}`).join(' ');
        return `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="3"/>`;
      }};

      svg += line('y_pred', '#2457d6');
      svg += line('y_obs', '#14845f');
      svg += `</svg>`;
      document.getElementById('calibration-container').innerHTML = svg;
    }}

    // 5. Importance SVG
    const imps = data.validation.top_features || [];
    if (imps.length > 0) {{
      const h = imps.length * 40 + 20, w = 600, left = 200;
      const maxScore = Math.max(...imps.map(i => i.score));
      let svg = `<svg viewBox="0 0 ${{w}} ${{h}}">`;
      imps.forEach((imp, i) => {{
        const y = i * 40 + 20;
        const barW = (imp.score / maxScore) * (w - left - 60);
        svg += `<text x="10" y="${{y+15}}" font-size="12" fill="var(--ink)">${{imp.feature}}</text>`;
        svg += `<rect x="${{left}}" y="${{y}}" width="${{barW}}" height="20" fill="var(--blue)" rx="4"/>`;
        svg += `<text x="${{left + barW + 10}}" y="${{y+15}}" font-size="12" fill="var(--muted)">${{imp.score.toFixed(3)}}</text>`;
      }});
      svg += `</svg>`;
      document.getElementById('importance-container').innerHTML = svg;
    }}

    // 6. Env Table
    const envTable = document.getElementById('env-table');
    Object.entries(data.environment).forEach(([k, v]) => {{
      const tr = document.createElement('tr');
      const tdK = document.createElement('td');
      const tdV = document.createElement('td');
      tdK.textContent = String(k);
      tdV.textContent = String(v);
      tr.appendChild(tdK);
      tr.appendChild(tdV);
      envTable.appendChild(tr);
    }});
  </script>
</body>
</html>"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
