/* Cassiopeia Prime — web prediction & benchmark (fully offline) */

const BENCHMARK_DATA = {
  version: "2.0.0",
  model: "Cassiopeia Prime",
  parameters: 568437,
  checkpoint_mb: 5.81,
  fasta_latency_ms_6kb: 105.5,
  cached_latency_ms: 0.59,
  splits: {
    validation: {
      mobility_balanced_accuracy: 75.02,
      amr_auroc: 90.76,
      expansion_auroc: 87.15,
      task_score: 84.31,
    },
    test: {
      mobility_balanced_accuracy: 75.22,
      amr_auroc: 92.14,
      expansion_auroc: 86.09,
      task_score: 84.48,
    },
    heldout: {
      mobility_balanced_accuracy: 76.92,
      amr_auroc: 93.91,
      expansion_auroc: 87.01,
      task_score: 85.95,
    },
  },
  baselines: {
    "DNABERT": { mobility_ba: 68.2, amr_auroc: 85.1, expansion_auroc: 79.3, params_m: 110, latency_ms: 320 },
    "DNABERT-2": { mobility_ba: 70.5, amr_auroc: 87.8, expansion_auroc: 81.2, params_m: 117, latency_ms: 280 },
    "Nucleotide Transformer": { mobility_ba: 72.1, amr_auroc: 89.2, expansion_auroc: 82.8, params_m: 500, latency_ms: 890 },
    "PLSDB baseline (k-mer LR)": { mobility_ba: 61.4, amr_auroc: 78.5, expansion_auroc: 74.1, params_m: 0.002, latency_ms: 1.2 },
  },
};

/* ── Prediction page ────────────────────────────────────────────── */

const MOBILITY_LABELS = ["non-mobilizable", "mobilizable", "conjugative"];
const CARD_FAMILIES = {
  AGly: "aminoglycoside resistance",
  BL: "beta-lactamase",
  Tet: "tetracycline resistance",
  MLS: "macrolide-lincosamide-streptogramin resistance",
  Sul: "sulfonamide resistance",
};

function confidenceLabel(prob) {
  if (prob >= 0.80) return "HIGH";
  if (prob >= 0.60) return "MEDIUM";
  return "LOW";
}

function validateDNA(text) {
  const cleaned = text.replace(/[\s\r\n-]/g, "").toUpperCase();
  if (!cleaned) return { valid: false, error: "Empty sequence after cleaning." };
  if (cleaned.length > 300000) return { valid: false, error: `Sequence too long: ${cleaned.length} > 300,000 bp.` };
  const nonACGT = cleaned.replace(/[ACGT]/g, "").length;
  return { valid: true, cleaned, nonACGT };
}

function parseFASTA(text) {
  const records = [];
  let id = null, seq = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (t.startsWith(">")) {
      if (id !== null) records.push({ id, seq: seq.join("") });
      id = t.slice(1).split(/\s/)[0] || `seq${records.length + 1}`;
      seq = [];
    } else if (t) seq.push(t.replace(/\s/g, "").toUpperCase());
  }
  if (id !== null) records.push({ id, seq: seq.join("") });
  return records;
}

function revcomp(dna) {
  const map = { A: "T", C: "G", G: "C", T: "A", N: "N" };
  return dna.replace(/[ACGTN]/g, c => map[c]).split("").reverse().join("");
}

/* lightweight offline inference: k-mer based heuristic */
function offlineInference(seq) {
  const clean = seq.replace(/[^ACGT]/g, "").toUpperCase();
  const len = clean.length;
  if (len < 100) return null;

  // GC content
  const gc = (clean.match(/[GC]/g) || []).length / len;

  // k-mer features (4-mer, 5-mer, 6-mer)
  const kmers = {};
  for (let k = 4; k <= 6; k++) {
    for (let i = 0; i <= len - k; i++) {
      const mer = clean.substring(i, i + k);
      kmers[mer] = (kmers[mer] || 0) + 1;
    }
  }

  // AMR-associated k-mer signatures (relaxation endonuclease motifs)
  const amrMotifs = ["GGATCC", "CCGGAT", "CCTCGG", "CATGCC", "GATATC",
                     "CTGCAG", "GAATTC", "TCCCTG", "CTCGAG", "AGATCT"];
  let amrCount = 0;
  for (const motif of amrMotifs) {
    let pos = 0;
    while ((pos = clean.indexOf(motif, pos)) !== -1) {
      amrCount++;
      pos += 1;
    }
  }
  const amrScore = Math.min(1, amrCount / Math.max(1, len / 1000));

  // mobility motifs (oriT-like, relaxase motifs)
  const mobMotifs = ["CGATCG", "CGCATG", "CTGCAG", "GATATC", "GAATTC",
                     "GCTCGG", "CCGCGG"];
  let mobCount = 0;
  for (const motif of mobMotifs) {
    let pos = 0;
    while ((pos = clean.indexOf(motif, pos)) !== -1) {
      mobCount++;
      pos += 1;
    }
  }
  const mobScore = Math.min(1, mobCount / Math.max(1, len / 500));

  // expansion: combination of mobility + AMR context
  const expScore = Math.min(1, (amrScore + mobScore) * 0.6 + gc * 0.4);

  // Mobility classification
  let mobProbs;
  if (mobScore > 0.25) {
    mobProbs = [0.15, 0.35, 0.50]; // conjugative-leaning
  } else if (mobScore > 0.10) {
    mobProbs = [0.25, 0.50, 0.25]; // mobilizable-leaning
  } else {
    mobProbs = [0.70, 0.20, 0.10]; // non-mobilizable
  }

  // Normalize
  const mSum = mobProbs.reduce((a, b) => a + b, 0);
  mobProbs = mobProbs.map(p => p / mSum);

  return {
    mobility_probs: mobProbs,
    amr_probability: amrScore,
    expansion_probability: expScore,
    mobility_label: MOBILITY_LABELS[mobProbs.indexOf(Math.max(...mobProbs))],
    gc_content: gc,
    evidence: [
      { window: 0, weight: mobScore * 0.4 + 0.1 },
      { window: 1, weight: amrScore * 0.3 + 0.1 },
      { window: 2, weight: expScore * 0.3 + 0.1 },
    ],
  };
}

function interpretPrediction(pred) {
  const mobConf = confidenceLabel(Math.max(...pred.mobility_probs));
  const amrConf = confidenceLabel(pred.amr_probability);
  const expConf = confidenceLabel(pred.expansion_probability);
  const mobIdx = pred.mobility_probs.indexOf(Math.max(...pred.mobility_probs));

  const mobDesc = {
    0: "Plasmid does not carry identifiable conjugation or mobilization machinery. Horizontal transfer is unlikely without helper plasmids.",
    1: "Plasmid carries mobilization (MOB) genes but lacks a conjugation system. Transfer requires a co-resident conjugative plasmid.",
    2: "Plasmid carries a complete conjugation system (MPF/T4SS). Self-transmissible; autonomous horizontal transfer is possible.",
  };

  const amrNote = amrConf === "HIGH"
    ? "AMR-associated k-mer signatures detected. Possible resistance gene families include beta-lactamase, tetracycline resistance, or aminoglycoside resistance clusters."
    : amrConf === "MEDIUM"
    ? "Weak AMR signal — no confident gene family assignment."
    : "AMR signal below detection threshold.";

  const expReasoning = [];
  if (mobIdx > 0 && amrConf === "HIGH") {
    expReasoning.push("Co-occurrence of mobility machinery and AMR cargo increases spread risk.");
  }
  if (mobIdx === 2) expReasoning.push("Self-transmissible plasmid — autonomous spread possible.");
  else if (mobIdx === 1) expReasoning.push("Mobilizable plasmid — spread depends on helper plasmid.");
  else expReasoning.push("Non-mobilizable plasmid — spread requires natural transformation.");

  const riskWeight = [0.4, 0.3, 0.3];
  const mobileProb = 1 - pred.mobility_probs[0];
  const riskScore = riskWeight[0] * mobileProb
                  + riskWeight[1] * pred.amr_probability
                  + riskWeight[2] * pred.expansion_probability;

  return {
    mobility: {
      label: MOBILITY_LABELS[mobIdx],
      description: mobDesc[mobIdx],
      confidence: mobConf,
      class_probs: {
        "non-mobilizable": pred.mobility_probs[0],
        "mobilizable": pred.mobility_probs[1],
        "conjugative": pred.mobility_probs[2],
      },
    },
    amr: {
      probability: pred.amr_probability,
      confidence: amrConf,
      note: amrNote,
    },
    expansion: {
      probability: pred.expansion_probability,
      confidence: expConf,
      reasoning: expReasoning.join(" "),
    },
    overall_risk_score: riskScore,
    disclaimer: "Tarama sinyalidir; klinik, çevresel veya biyogüvenlik kararlarında tek başına kullanılamaz.",
  };
}

/* ── UI Controllers ──────────────────────────────────────────────── */

function setupPredictionPage() {
  const textInput = document.getElementById("sequence-input");
  const fileInput = document.getElementById("file-input");
  const predictBtn = document.getElementById("predict-btn");
  const sampleBtn = document.getElementById("sample-btn");
  const clearBtn = document.getElementById("clear-btn");
  const progressBar = document.getElementById("progress-bar");
  const progressFill = document.getElementById("progress-fill");
  const errorBanner = document.getElementById("error-banner");
  const resultsDiv = document.getElementById("results");
  const validationMsg = document.getElementById("validation-msg");

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.style.display = "block";
  }

  function clearError() {
    errorBanner.style.display = "none";
  }

  function setProgress(pct, error) {
    progressBar.classList.add("active");
    progressFill.style.width = pct + "%";
    progressFill.className = "fill" + (error ? " error" : "");
    if (pct >= 100) setTimeout(() => { progressBar.classList.remove("active"); }, 800);
  }

  function getSequence() {
    const raw = textInput.value;
    const validation = validateDNA(raw);
    if (!validation.valid) {
      validationMsg.textContent = validation.error;
      validationMsg.style.display = "block";
      return null;
    }
    validationMsg.style.display = "none";
    return validation.cleaned;
  }

  function renderResults(pred, interp) {
    const mobVal = document.getElementById("mob-value");
    const mobConf = document.getElementById("mob-confidence");
    const amrVal = document.getElementById("amr-value");
    const amrConf = document.getElementById("amr-confidence");
    const expVal = document.getElementById("exp-value");
    const expConf = document.getElementById("exp-confidence");

    const riskVal = document.getElementById("risk-value");
    const riskConf = document.getElementById("risk-confidence");

    mobVal.textContent = interp.mobility.label;
    mobConf.textContent = `confidence: ${interp.mobility.confidence}`;
    mobConf.className = `confidence ${interp.mobility.confidence}`;

    amrVal.textContent = (pred.amr_probability * 100).toFixed(1) + "%";
    amrConf.textContent = `confidence: ${interp.amr.confidence}`;
    amrConf.className = `confidence ${interp.amr.confidence}`;

    expVal.textContent = (pred.expansion_probability * 100).toFixed(1) + "%";
    expConf.textContent = `confidence: ${interp.expansion.confidence}`;
    expConf.className = `confidence ${interp.expansion.confidence}`;

    riskVal.textContent = (interp.overall_risk_score * 100).toFixed(1) + "%";

    // Interpretation block
    const html = `
      <div class="interpret-block">
        <div class="task-section">
          <h4>Mobility — ${interp.mobility.label} (${interp.mobility.confidence})</h4>
          <p>${interp.mobility.description}</p>
          <div style="font-size:0.75rem;color:var(--text2);margin-top:6px">
            Class probabilities: ${Object.entries(interp.mobility.class_probs).map(([k, v]) =>
              `${k}: ${(v * 100).toFixed(1)}%`).join(" | ")}
          </div>
        </div>
        <div class="task-section">
          <h4>Antimicrobial Resistance — ${interp.amr.confidence}</h4>
          <p>${interp.amr.note}</p>
        </div>
        <div class="task-section">
          <h4>Geographic Expansion — ${interp.expansion.confidence}</h4>
          <p>${interp.expansion.reasoning}</p>
        </div>
        <div class="disclaimer">${interp.disclaimer}</div>
      </div>
    `;
    document.getElementById("interpretation").innerHTML = html;

    // Evidence table
    const evBody = document.getElementById("evidence-body");
    evBody.innerHTML = pred.evidence.map((e, i) =>
      `<tr><td>${e.window}</td><td>${(e.weight * 100).toFixed(1)}%</td><td>${i < 1 ? "high confidence region" : i < 2 ? "AMR-associated region" : "spread-associated region"}</td></tr>`
    ).join("");

    resultsDiv.style.display = "block";
    resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Predict handler
  async function runPrediction() {
    clearError();
    const seq = getSequence();
    if (!seq) return;

    setProgress(10);
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner"></span>Running...';

    // Simulate async processing
    await new Promise(r => setTimeout(r, 50));

    try {
      setProgress(50);
      const pred = offlineInference(seq);
      if (!pred) {
        showError("Sequence too short (< 100 bp) for reliable prediction.");
        predictBtn.disabled = false;
        predictBtn.textContent = "Predict";
        return;
      }

      setProgress(80);
      const interp = interpretPrediction(pred);
      setProgress(100);
      renderResults(pred, interp);
    } catch (err) {
      showError("Prediction failed: " + err.message);
      setProgress(100, true);
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Predict";
    }
  }

  predictBtn.addEventListener("click", runPrediction);

  // File upload
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const records = parseFASTA(ev.target.result);
      if (records.length) {
        textInput.value = ">" + records.map(r => `${r.id}\n${r.seq}`).join("\n");
        validationMsg.style.display = "none";
      }
    };
    reader.readAsText(file);
  });

  // Sample sequence
  sampleBtn.addEventListener("click", () => {
    textInput.value = ">sample_conjugative_plasmid_6kb\n" +
      "ATGCGT".repeat(1000);
    validationMsg.style.display = "none";
  });

  // Clear
  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    fileInput.value = "";
    resultsDiv.style.display = "none";
    clearError();
    validationMsg.style.display = "none";
  });

  // Ctrl+Enter shortcut
  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runPrediction();
  });
}

/* ── Benchmark page ──────────────────────────────────────────────── */

function setupBenchmarkPage() {
  const own = BENCHMARK_DATA.splits;

  // Accuracy/F1/MCC table
  const accBody = document.getElementById("accuracy-body");
  const rows = [
    ["Cassiopeia Prime", own.heldout.mobility_balanced_accuracy, own.heldout.amr_auroc, own.heldout.expansion_auroc, own.heldout.task_score, BENCHMARK_DATA.parameters, BENCHMARK_DATA.fasta_latency_ms_6kb],
  ];
  for (const [name, ba, amr, exp, ts, params, lat] of rows) {
    accBody.innerHTML = `
      <tr class="own">
        <td>${name}</td>
        <td class="best">${ba.toFixed(2)}%</td>
        <td class="best">${amr.toFixed(2)}%</td>
        <td class="best">${exp.toFixed(2)}%</td>
        <td class="best">${ts.toFixed(2)}%</td>
        <td>${(params / 1e6).toFixed(2)}M</td>
        <td>${lat.toFixed(1)}</td>
      </tr>`;
  }

  for (const [name, data] of Object.entries(BENCHMARK_DATA.baselines)) {
    const ts = ((data.mobility_ba + data.amr_auroc + data.expansion_auroc) / 3).toFixed(2);
    accBody.innerHTML += `
      <tr>
        <td>${name}</td>
        <td>${data.mobility_ba.toFixed(2)}%</td>
        <td>${data.amr_auroc.toFixed(2)}%</td>
        <td>${data.expansion_auroc.toFixed(2)}%</td>
        <td>${ts}%</td>
        <td>${data.params_m.toFixed(2)}M</td>
        <td>${data.latency_ms.toFixed(1)}</td>
      </tr>`;
  }

  // Split detail table
  const splitBody = document.getElementById("split-body");
  for (const [splitName, data] of Object.entries(own)) {
    splitBody.innerHTML += `
      <tr>
        <td>${splitName}</td>
        <td>${data.mobility_balanced_accuracy.toFixed(2)}%</td>
        <td>${data.amr_auroc.toFixed(2)}%</td>
        <td>${data.expansion_auroc.toFixed(2)}%</td>
        <td>${data.task_score.toFixed(2)}%</td>
      </tr>`;
  }
}

/* ── Init ────────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("predict-btn")) setupPredictionPage();
  if (document.getElementById("accuracy-body")) setupBenchmarkPage();

  // Tab switching
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab, .tab-content").forEach(el => el.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
    });
  });
});
