const fmtPct = (v) => `${(v * 100).toFixed(2)}%`;
const fmtInt = (v) => new Intl.NumberFormat("en-US").format(v);

function metricCard(name, split) {
  if (!("task_score" in split)) {
    const rows = [
      ["False mobile", split.false_mobile_rate],
      ["False AMR", split.false_amr_rate],
      ["False expansion", split.false_expansion_rate],
      ["Mean risk", split.risk_mean],
    ];
    return `<article class="metric-card stress-card">
      <span>${name}</span>
      <h3>${fmtPct(split.false_expansion_rate ?? 0)}</h3>
      ${rows.map(([label, value]) => `<div class="metric-row">
        <b>${label}</b>
        <div class="bar"><i style="--value:${Math.round(value * 100)}%"></i></div>
        <em>${fmtPct(value)}</em>
      </div>`).join("")}
    </article>`;
  }
  const rows = [
    ["Mobility BA", split.mobility_balanced_accuracy],
    ["AMR AUROC", split.amr_auroc],
    ["Exp. AUROC", split.expansion_auroc],
    ["Task score", split.task_score],
  ];
  return `<article class="metric-card">
    <span>${name}</span>
    <h3>${fmtPct(split.task_score)}</h3>
    ${rows.map(([label, value]) => `<div class="metric-row">
      <b>${label}</b>
      <div class="bar"><i style="--value:${Math.round(value * 100)}%"></i></div>
      <em>${fmtPct(value)}</em>
    </div>`).join("")}
  </article>`;
}

async function loadBenchmark() {
  let data;
  try {
    const res = await fetch("benchmark.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch {
    document.querySelector("#metrics").innerHTML = "<p class='error-msg'>Benchmark yüklenemedi.</p>";
    return;
  }
  document.querySelector("#params").textContent = fmtInt(data.parameters ?? data.params ?? 0);
  document.querySelector("#checkpoint").textContent = `${(data.checkpoint_mb ?? 0).toFixed(2)} MB`;
  const latency = data.cached_latency_ms ?? data.latency_ms_per_cached_sequence ?? 0;
  document.querySelector("#latency").textContent = `${latency.toFixed(2)} ms`;
  const fastaLatency = data.fasta_latency_ms_6kb ?? 0;
  document.querySelector("#fasta-latency").textContent = `${fastaLatency.toFixed(2)} ms`;
  const hk = data.splits?.heldout ?? data.splits?.heldout_test;
  if (hk) document.querySelector("#task-score").textContent = fmtPct(hk.task_score ?? 0);
  const s = data.splits || {};
  const order = [
    ["Validation", s.validation ?? s.val],
    ["Test", s.test],
    ["Held-out", s.heldout ?? s.heldout_test],
    ["Non-plasmid stress", s.nonplasmid_control],
  ];
  document.querySelector("#metrics").innerHTML = order
    .filter(([, d]) => d)
    .map(([n, d]) => metricCard(n, d)).join("");
}

function updateSequenceStats() {
  const text = document.querySelector("#fasta").value;
  const seq = text.split("\n").filter((l) => !l.startsWith(">")).join("").toUpperCase().replace(/[^A-Z]/g, "");
  const gc = (seq.match(/[GC]/g) || []).length;
  const n = (seq.match(/[^ACGT]/g) || []).length;
  document.querySelector("#seq-len").textContent = `${seq.length} bp`;
  document.querySelector("#seq-gc").textContent = seq.length ? `${(100 * gc / seq.length).toFixed(1)}%` : "0.0%";
  document.querySelector("#seq-n").textContent = n;
  return seq;
}

async function runInference() {
  const btn = document.querySelector("#infer-btn");
  const out = document.querySelector("#infer-out");
  const seq = document.querySelector("#fasta").value.trim();
  if (!seq) return;
  const dna = seq.split("\n").filter((l) => !l.startsWith(">")).join("").replace(/\s/g, "").toUpperCase();
  if (!dna) return;
  btn.disabled = true;
  btn.textContent = "Tahmin ediliyor…";
  out.innerHTML = "";
  let apiUrl = document.querySelector("#api-url").value || window.location.origin;

  // Auto-detect if running from filesystem or docs/ directly
  if (apiUrl === window.location.origin && window.location.protocol === "file:") {
    apiUrl = "http://localhost:8000";
  }

  try {
    const res = await fetch(`${apiUrl.replace(/\/+$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence_id: "query", dna }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
    }
    const p = await res.json();
    const risk = (p.risk_score ?? 0);
    let riskClass = "risk-low";
    if (risk >= 0.7) riskClass = "risk-high";
    else if (risk >= 0.4) riskClass = "risk-mid";
    const mobLabel = ["Non-mobilizable", "Mobilizable", "Conjugative"];
    const mobIdx = p.mobility_probs ? p.mobility_probs.indexOf(Math.max(...p.mobility_probs)) : 0;
    out.innerHTML = `<div class="infer-result ${riskClass}">
      <div class="risk-gauge"><strong>Risk</strong><span>${(risk * 100).toFixed(1)}%</span></div>
      <div class="infer-grid">
        <div><span>Mobility</span><strong>${mobLabel[mobIdx]}</strong></div>
        <div><span>AMR</span><strong>${(p.amr_probability * 100).toFixed(1)}%</strong></div>
        <div><span>Expansion</span><strong>${(p.expansion_probability * 100).toFixed(1)}%</strong></div>
      </div>
      <div class="infer-evidence">
        <details><summary>Task-specific evidence</summary>
          <pre>${JSON.stringify({
            mobility_probs: p.mobility_probs,
            top_mobility_windows: p.top_mobility_windows,
            top_amr_windows: p.top_amr_windows,
            top_expansion_windows: p.top_expansion_windows
          }, null, 2)}</pre>
        </details>
      </div>
    </div>`;
  } catch (err) {
    out.innerHTML = `<div class="infer-error">
      <p>API bağlantı hatası: ${err.message}</p>
      <p class="infer-hint">Sunucu çalışıyor mu kontrol et: <code>dna-sentinel serve --checkpoint ...</code></p>
    </div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Tahmin Et";
  }
}

document.querySelector("#fasta").addEventListener("input", updateSequenceStats);
document.querySelector("#infer-btn")?.addEventListener("click", runInference);

updateSequenceStats();
loadBenchmark();
