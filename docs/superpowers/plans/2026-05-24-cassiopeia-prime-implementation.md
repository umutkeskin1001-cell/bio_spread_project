# Cassiopeia Prime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one maximally optimized compact Cassiopeia Prime model under 1M trainable parameters, focused on biologically correct invariance, stronger training, and task-specific evidence.

**Architecture:** Preserve the current Cassiopeia pipeline and add only high-leverage changes: configurable capacity, optional CPPE, optional window-level motif convolution, stricter mask semantics, task-specific evidence outputs, equal-weight scoring, balanced sampling, consistency regularization, and a compact benchmark report. Avoid model families, distillation, large refactors, and speculative attention layers until this single model is saturated.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, scikit-learn, Click, PyYAML, pytest, existing `dna_sentinel` package.

---

## Operating Principles

- Optimize under a hard budget: Prime must stay at or below 1,000,000 trainable parameters.
- Keep default Cassiopeia backward-compatible: existing tests and old default config should remain small and loadable.
- Prime is activated by config, not by changing defaults aggressively.
- Do not split `src/dna_sentinel/train.py` into a package during this pass.
- Implement ablation-friendly flags so CPPE, window convolution, balanced sampling, and consistency can be switched independently.
- Every task adds tests before implementation.
- Commit after each task.

---

## Final Technical Decisions

### Model config for Prime

Use this initial Prime budget. If exact parameter count exceeds 1,000,000 after implementation, reduce `frp_out_dim` from `384` to `320`; do not reduce the biological modules first.

```yaml
model:
  n_canonical_features: 2728
  n_structural_features: 19
  hidden_dim: 160
  frp_out_dim: 384
  n_layers: 3
  lora_rank: 12
  adapter_rank: 16
  n_evidence_heads: 1
  drop_path_rate: 0.12
  aux_loss_weight: 0.25
  dropout: 0.12
  max_windows: 28
  expansion_classes: 1
  amr_classes: 1
  label_smoothing: 0.08
  use_scale_embedding: false
  use_cppe: true
  use_window_conv: true
  window_conv_kernel: 5
```

### Training config for Prime

```yaml
training:
  epochs: 120
  batch_size: 32
  lr: 0.0003
  backbone_lr: 0.00025
  head_lr: 0.00035
  min_lr: 0.00001
  warmup_epochs: 6
  weight_decay: 0.05
  patience: 30
  gradient_accumulation_steps: 2
  focal_loss_gamma: 2.0
  mixup_alpha: 0.0
  dropout: 0.12
  balanced_sampling: true
  consistency_weight: 0.08
  consistency_temperature: 1.0
  score_mode: equal
  artifact_dir: artifacts/cassiopeia_prime
```

Rationale: consistency regularization and mixup both regularize probability geometry. The first Prime run should not combine them. Use consistency first because it encodes plasmid biology directly.

### Self-critique gates

- If Prime only improves validation but not test or heldout, the model is overfitting; reduce hidden capacity or consistency weight before adding modules.
- If CPPE worsens circular-shift drift, its coordinate computation is wrong or scale embeddings are fighting it.
- If window convolution improves AMR but hurts mobility, the convolution is likely smoothing scale boundaries; reduce kernel to 3 or place convolution after context gate.
- If balanced sampling improves mobility but harms AMR/expansion calibration, lower sampling strength instead of changing task weights.
- If consistency reduces AUROC while improving drift, consistency weight is too high or transformed features are not label-preserving.
- If parameter budget is exceeded, reduce `frp_out_dim` first, then `hidden_dim`; do not remove evidence or invariance modules.

---

## File Structure

### Modify

- `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/model.py`  
  Add Prime config fields, CPPE module, window convolution module, adapter rank config, mask cleanup, task-specific evidence outputs.

- `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/train.py`  
  Add equal-weight score, balanced sampling, optional consistency loss using cached transformed features.

- `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/features.py`  
  Add deterministic consistency feature cache generation for train split.

- `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/utils.py`  
  Add `circular_shift`, richer `Prediction`, non-plasmid false-positive metrics, and task-specific top evidence extraction.

- `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/cli.py`  
  Load optional consistency feature cache and add a compact `benchmark` command with report output.

- `/Users/umut/Projeler/bio_spread_project/config/cassiopeia_prime.yaml`  
  New focused Prime config.

- `/Users/umut/Projeler/bio_spread_project/tests/test_model.py`  
  Tests for config fields, CPPE/window conv shapes, mask behavior, parameter budget, evidence outputs.

- `/Users/umut/Projeler/bio_spread_project/tests/test_train.py`  
  Tests for equal score, balanced weights, consistency loss.

- `/Users/umut/Projeler/bio_spread_project/tests/test_features.py`  
  Tests for deterministic consistency transform feature generation.

- `/Users/umut/Projeler/bio_spread_project/tests/test_cli_predict.py`  
  Tests for prediction JSON containing task-specific evidence.

### Create

- `/Users/umut/Projeler/bio_spread_project/tests/test_benchmark.py`  
  Tests for report scoring and non-plasmid false-positive summaries.

---

## Task 1: Add Prime Config and Parameter-Budget Tests

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/model.py:11-26`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_model.py`
- Create: `/Users/umut/Projeler/bio_spread_project/config/cassiopeia_prime.yaml`

- [ ] **Step 1: Write failing tests for new config fields and Prime budget**

Add this to `tests/test_model.py` inside `TestCassiopeiaConfig`:

```python
def test_prime_config_fields_are_parsed(self):
    yaml_dict = {
        "model": {
            "hidden_dim": 160,
            "frp_out_dim": 384,
            "n_layers": 3,
            "lora_rank": 12,
            "adapter_rank": 16,
            "use_scale_embedding": False,
            "use_cppe": True,
            "use_window_conv": True,
            "window_conv_kernel": 5,
        }
    }
    cfg = CassiopeiaConfig.from_yaml(yaml_dict)
    assert cfg.hidden_dim == 160
    assert cfg.frp_out_dim == 384
    assert cfg.n_layers == 3
    assert cfg.lora_rank == 12
    assert cfg.adapter_rank == 16
    assert cfg.use_scale_embedding is False
    assert cfg.use_cppe is True
    assert cfg.use_window_conv is True
    assert cfg.window_conv_kernel == 5
```

Add this to `TestCassiopeiaSmall`:

```python
def test_default_model_stays_compact(self):
    model = Cassiopeia()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count < 500_000


def test_prime_model_stays_under_one_million_parameters(self):
    model = Cassiopeia(CassiopeiaConfig(
        hidden_dim=160,
        frp_out_dim=384,
        n_layers=3,
        lora_rank=12,
        adapter_rank=16,
        use_scale_embedding=False,
        use_cppe=True,
        use_window_conv=True,
        window_conv_kernel=5,
    ))
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count <= 1_000_000
```

- [ ] **Step 2: Run config tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_model.py::TestCassiopeiaConfig::test_prime_config_fields_are_parsed tests/test_model.py::TestCassiopeiaSmall::test_prime_model_stays_under_one_million_parameters -v
```

Expected: FAIL because `adapter_rank`, `use_scale_embedding`, `use_cppe`, `use_window_conv`, and `window_conv_kernel` are not config fields yet.

- [ ] **Step 3: Add config fields**

Modify `CassiopeiaConfig` in `src/dna_sentinel/model.py`:

```python
@dataclass(frozen=True)
class CassiopeiaConfig:
    n_canonical_features: int = 2728
    n_structural_features: int = 19
    hidden_dim: int = 128
    frp_out_dim: int = 256
    n_layers: int = 2
    lora_rank: int = 8
    adapter_rank: int = 0
    n_evidence_heads: int = 1
    drop_path_rate: float = 0.1
    aux_loss_weight: float = 0.3
    dropout: float = 0.15
    max_windows: int = 28
    expansion_classes: int = 1
    amr_classes: int = 1
    label_smoothing: float = 0.1
    use_scale_embedding: bool = True
    use_cppe: bool = False
    use_window_conv: bool = False
    window_conv_kernel: int = 5
```

- [ ] **Step 4: Use configurable adapter rank**

Modify adapter rank selection in `Cassiopeia.__init__`:

```python
r = cfg.adapter_rank if cfg.adapter_rank > 0 else max(1, h // 16)
```

- [ ] **Step 5: Create Prime config**

Create `/Users/umut/Projeler/bio_spread_project/config/cassiopeia_prime.yaml`:

```yaml
data:
  fasta_path: data/project_inputs/raw/plsdb_sequences.fasta
  backbones_tsv: data/project_inputs/silver/plasmid_backbones.tsv
  amr_tsv: data/project_inputs/silver/plasmid_amr_consensus.tsv
  out_dir: data/dna_sentinel
  limit: 2048
  min_len: 1000
  max_len: 300000
  seed: 42
  expansion_country_threshold: 15

model:
  n_canonical_features: 2728
  n_structural_features: 19
  hidden_dim: 160
  frp_out_dim: 384
  n_layers: 3
  lora_rank: 12
  adapter_rank: 16
  n_evidence_heads: 1
  drop_path_rate: 0.12
  aux_loss_weight: 0.25
  dropout: 0.12
  max_windows: 28
  expansion_classes: 1
  amr_classes: 1
  label_smoothing: 0.08
  use_scale_embedding: false
  use_cppe: true
  use_window_conv: true
  window_conv_kernel: 5

features:
  window_sizes: [512, 2048, 8192]
  strides: [256, 1024, 4096]
  max_windows: [16, 8, 4]
  build_consistency_cache: true

training:
  epochs: 120
  batch_size: 32
  lr: 0.0003
  backbone_lr: 0.00025
  head_lr: 0.00035
  min_lr: 0.00001
  warmup_epochs: 6
  weight_decay: 0.05
  patience: 30
  gradient_accumulation_steps: 2
  focal_loss_gamma: 2.0
  mixup_alpha: 0.0
  dropout: 0.12
  balanced_sampling: true
  consistency_weight: 0.08
  consistency_temperature: 1.0
  score_mode: equal
  artifact_dir: artifacts/cassiopeia_prime
```

- [ ] **Step 6: Run tests**

Run:

```bash
python3 -m pytest tests/test_model.py::TestCassiopeiaConfig::test_prime_config_fields_are_parsed tests/test_model.py::TestCassiopeiaSmall::test_default_model_stays_compact tests/test_model.py::TestCassiopeiaSmall::test_prime_model_stays_under_one_million_parameters -v
```

Expected: PASS. This task enforces the hard upper budget only; Task 2 will raise the actual count by adding CPPE and window-conv modules.

- [ ] **Step 7: Commit**

```bash
git add src/dna_sentinel/model.py tests/test_model.py config/cassiopeia_prime.yaml
git commit -m "feat: add Cassiopeia Prime config budget"
```

---

## Task 2: Add CPPE and Window Motif Convolution

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/model.py:52-167`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_model.py`

- [ ] **Step 1: Write failing tests for CPPE and window convolution**

Add imports in `tests/test_model.py`:

```python
from dna_sentinel.model import CircularPositionEncoding, WindowMotifConv
```

Add to `TestBlocks`:

```python
def test_cppe_shape_and_mask_zeroing(self):
    cppe = CircularPositionEncoding(32)
    x = torch.zeros(2, 6, 32)
    mask = torch.tensor([[True, True, True, False, False, False],
                         [True, True, True, True, True, True]])
    scale_ids = torch.tensor([[0, 0, 1, 1, 2, 2],
                              [0, 0, 1, 1, 2, 2]])
    out = cppe(x, mask, scale_ids)
    assert out.shape == x.shape
    assert torch.all(out[0, 3:] == 0)
    assert out[1].abs().sum() > 0


def test_window_motif_conv_shape_and_mask_zeroing(self):
    conv = WindowMotifConv(hidden_dim=32, kernel_size=5, dropout=0.0)
    x = torch.randn(2, 8, 32)
    mask = torch.tensor([[True, True, True, True, False, False, False, False],
                         [True, True, True, True, True, True, True, True]])
    out = conv(x, mask)
    assert out.shape == x.shape
    assert torch.all(out[0, 4:] == 0)
    assert torch.isfinite(out).all()
```

Add to `TestCassiopeiaSmall`:

```python
def test_prime_forward_with_cppe_and_window_conv(self):
    model = Cassiopeia(CassiopeiaConfig(
        hidden_dim=64,
        n_canonical_features=100,
        frp_out_dim=80,
        n_layers=2,
        max_windows=12,
        lora_rank=4,
        adapter_rank=8,
        use_scale_embedding=False,
        use_cppe=True,
        use_window_conv=True,
        window_conv_kernel=3,
    ))
    features = torch.randn(3, 12, 100)
    mask = torch.ones(3, 12, dtype=torch.bool)
    struct = torch.randn(3, 12, 19)
    scale_ids = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4] * 3)
    out = model(features, mask, struct_features=struct, scale_ids=scale_ids)
    assert out["mobility_logits"].shape == (3, 3)
    assert out["amr_logits"].shape == (3,)
    assert out["expansion_logits"].shape == (3,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_model.py::TestBlocks::test_cppe_shape_and_mask_zeroing tests/test_model.py::TestBlocks::test_window_motif_conv_shape_and_mask_zeroing tests/test_model.py::TestCassiopeiaSmall::test_prime_forward_with_cppe_and_window_conv -v
```

Expected: FAIL because the classes do not exist.

- [ ] **Step 3: Implement `CircularPositionEncoding`**

Add below `ContextGate` or above `CassiopeiaEncoder` in `src/dna_sentinel/model.py`:

```python
class CircularPositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, scale_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, w, _ = x.shape
        dev = x.device
        if scale_ids is None:
            pos = torch.arange(w, device=dev, dtype=x.dtype).unsqueeze(0).expand(b, -1)
            denom = torch.full((b, 1), max(1, w), device=dev, dtype=x.dtype)
            phase = 2.0 * math.pi * pos / denom
            scale_norm = torch.zeros_like(phase)
        else:
            sid = scale_ids.to(dev)
            pos = torch.zeros(b, w, device=dev, dtype=x.dtype)
            denom = torch.ones(b, w, device=dev, dtype=x.dtype)
            for scale in range(int(sid.max().item()) + 1 if sid.numel() else 1):
                sm = sid == scale
                counts = sm.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
                ranks = sm.to(dtype=x.dtype).cumsum(dim=1) - 1.0
                pos = torch.where(sm, ranks, pos)
                denom = torch.where(sm, counts.expand_as(denom), denom)
            phase = 2.0 * math.pi * pos / denom.clamp_min(1.0)
            scale_norm = sid.to(dtype=x.dtype) / max(1.0, float(int(sid.max().item()) if sid.numel() else 1))
        coords = torch.stack([torch.sin(phase), torch.cos(phase), scale_norm], dim=-1)
        return self.proj(coords) * mask.unsqueeze(-1).to(dtype=x.dtype)
```

- [ ] **Step 4: Implement `WindowMotifConv`**

Add below `CircularPositionEncoding`:

```python
class WindowMotifConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("window_conv_kernel must be odd")
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                   padding=kernel_size // 2, groups=hidden_dim)
        self.pointwise = nn.Linear(hidden_dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mf = mask.unsqueeze(-1).to(dtype=x.dtype)
        y = self.norm(x) * mf
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        gate, val = self.pointwise(y).chunk(2, dim=-1)
        y = self.drop(torch.sigmoid(gate) * F.gelu(val))
        return (x + y) * mf
```

- [ ] **Step 5: Wire modules into `CassiopeiaEncoder`**

Modify `CassiopeiaEncoder.__init__`:

```python
self.scale_embed = nn.Embedding(3, h) if cfg.use_scale_embedding else None
self.cppe = CircularPositionEncoding(h) if cfg.use_cppe else None
self.window_conv = (WindowMotifConv(h, cfg.window_conv_kernel, cfg.dropout)
                    if cfg.use_window_conv else None)
```

Modify `CassiopeiaEncoder.forward` after structural fusion and before `context_gate`:

```python
if self.scale_embed is not None and scale_ids is not None:
    x = x + self.scale_embed(scale_ids)
if self.cppe is not None:
    x = x + self.cppe(x, mask, scale_ids)
x = x * mf
if self.window_conv is not None:
    x = self.window_conv(x, mask)
x = self.context_gate(x, mask) * mf
```

Modify mixer loop to reapply mask:

```python
for mixer in self.mixers:
    x_m = mixer(x, mask) if self.drop_path_rate == 0 or not self.training else (
        x + (x.new_empty(x.shape[0], 1, 1).bernoulli_(keep_prob) / keep_prob) * (mixer(x, mask) - x))
    x = x_m * mf
    aux_features.append(x)
```

- [ ] **Step 6: Strengthen `GLUMixer` mask output**

Modify the return path in `GLUMixer.forward`:

```python
r = x
u, v = self.c_w1(self.c_norm(x)).chunk(2, dim=-1)
out = r + self.c_w2(self.c_drop(u * v))
return out if mask is None else out * mask.unsqueeze(-1).to(dtype=out.dtype)
```

- [ ] **Step 7: Run tests**

Run:

```bash
python3 -m pytest tests/test_model.py::TestBlocks::test_cppe_shape_and_mask_zeroing tests/test_model.py::TestBlocks::test_window_motif_conv_shape_and_mask_zeroing tests/test_model.py::TestCassiopeiaSmall::test_prime_forward_with_cppe_and_window_conv tests/test_model.py::TestCassiopeiaSmall::test_prime_model_stays_under_one_million_parameters -v
```

Expected: PASS. If parameter count exceeds 1,000,000, set Prime config/test `frp_out_dim=320` and rerun.

- [ ] **Step 8: Commit**

```bash
git add src/dna_sentinel/model.py tests/test_model.py config/cassiopeia_prime.yaml
git commit -m "feat: add CPPE and window motif convolution"
```

---

## Task 3: Return Task-Specific Evidence Without Breaking Checkpoints

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/model.py:88-238`
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/utils.py:126-176`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_model.py`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_cli_predict.py`

- [ ] **Step 1: Write failing evidence-output test**

Add to `TestCassiopeiaSmall`:

```python
def test_forward_returns_task_specific_evidence(self, model, batch):
    out = model(batch["features"], batch["masks"])
    assert out["mobility_evidence"].shape == batch["masks"].shape
    assert out["amr_evidence"].shape == batch["masks"].shape
    assert out["expansion_evidence"].shape == batch["masks"].shape
    assert torch.equal(out["mob_evidence"], out["mobility_evidence"])
```

Add to `TestInference`:

```python
def test_prediction_exposes_task_specific_windows(self):
    from dna_sentinel.utils import predict_one
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256,
                                          max_windows=28))
    pred = predict_one(model, "test", "ACGT" * 50)
    assert isinstance(pred.top_windows, list)
    assert isinstance(pred.top_mobility_windows, list)
    assert isinstance(pred.top_amr_windows, list)
    assert isinstance(pred.top_expansion_windows, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_model.py::TestCassiopeiaSmall::test_forward_returns_task_specific_evidence tests/test_model.py::TestInference::test_prediction_exposes_task_specific_windows -v
```

Expected: FAIL because the keys and dataclass fields do not exist.

- [ ] **Step 3: Return all evidence tensors from existing evidence heads**

Do not rename evidence parameters. Preserve checkpoint compatibility by reusing the existing `self.heads` module list.

Modify `MultiQueryEvidencePool.forward`:

```python
def forward(self, x_mob: torch.Tensor, x_amr: torch.Tensor, x_exp: torch.Tensor, mask: torch.Tensor):
    w = torch.softmax(self.mob_w, dim=0)
    all_scores = torch.stack([h(x_mob).squeeze(-1) for h in self.heads], dim=0)
    all_scores = all_scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(all_scores, dim=-1)
    mob_ctx = (w[:, None, None] * attn[:, :, :, None] * x_mob).sum(dim=(0, 2))
    mob_scores = all_scores[0]

    w = torch.softmax(self.amr_w, dim=0)
    all_scores = torch.stack([h(x_amr).squeeze(-1) for h in self.heads], dim=0)
    all_scores = all_scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(all_scores, dim=-1)
    amr_ctx = (w[:, None, None] * attn[:, :, :, None] * x_amr).sum(dim=(0, 2))
    amr_scores = all_scores[0]

    w = torch.softmax(self.exp_w, dim=0)
    all_scores = torch.stack([h(x_exp).squeeze(-1) for h in self.heads], dim=0)
    all_scores = all_scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(all_scores, dim=-1)
    exp_ctx = (w[:, None, None] * attn[:, :, :, None] * x_exp).sum(dim=(0, 2))
    exp_scores = all_scores[0]

    evidence = {
        "mobility_evidence": mob_scores,
        "amr_evidence": amr_scores,
        "expansion_evidence": exp_scores,
    }
    return (mob_ctx, amr_ctx, exp_ctx), evidence
```

Modify `forward_from_encoder`:

```python
(mob_ctx, amr_ctx, exp_ctx), evidence = self.evidence(x_mob, x_amr, x_exp, mask)
```

Modify result dict:

```python
result = {
    "mobility_logits": mob_logits,
    "amr_logits": amr_logits,
    "expansion_logits": exp_logits,
    "mobility_evidence": evidence["mobility_evidence"],
    "amr_evidence": evidence["amr_evidence"],
    "expansion_evidence": evidence["expansion_evidence"],
    "mob_evidence": evidence["mobility_evidence"],
}
```

- [ ] **Step 4: Extend `Prediction` dataclass**

Modify `Prediction` in `src/dna_sentinel/utils.py`:

```python
@dataclass(frozen=True)
class Prediction:
    sequence_id: str
    mobility_probs: list[float]
    amr_probability: float
    expansion_probability: float
    risk_score: float
    top_windows: list[dict[str, float]]
    top_mobility_windows: list[dict[str, float]]
    top_amr_windows: list[dict[str, float]]
    top_expansion_windows: list[dict[str, float]]
```

Add helper above `predict_batch`:

```python
def _top_evidence(scores: torch.Tensor, mask: torch.Tensor, top_k: int) -> list[dict[str, float]]:
    k = min(top_k, int(mask.sum().item()))
    if k <= 0:
        return []
    top = torch.topk(scores + (~mask).float() * (-1e9), k=k).indices
    return [{"window": float(i), "weight": float(scores[i])} for i in top.tolist() if bool(mask[i])]
```

Modify prediction assembly:

```python
mob_ev = out.get("mobility_evidence", out["mob_evidence"]).cpu()
amr_ev = out.get("amr_evidence", out["mob_evidence"]).cpu()
exp_ev = out.get("expansion_evidence", out["mob_evidence"]).cpu()
```

Then in loop:

```python
mob_wins = _top_evidence(mob_ev[idx], am[idx], top_k)
amr_wins = _top_evidence(amr_ev[idx], am[idx], top_k)
exp_wins = _top_evidence(exp_ev[idx], am[idx], top_k)
preds.append(Prediction(seq_id, mob[idx].tolist(), float(amr[idx]),
                         exp_val, risk, mob_wins, mob_wins, amr_wins, exp_wins))
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_model.py::TestCassiopeiaSmall::test_forward_returns_task_specific_evidence tests/test_model.py::TestInference::test_prediction_exposes_task_specific_windows tests/test_cli_predict.py::test_cli_predict_returns_json_for_fasta -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dna_sentinel/model.py src/dna_sentinel/utils.py tests/test_model.py tests/test_cli_predict.py
git commit -m "feat: expose task-specific evidence windows"
```

---

## Task 4: Use Equal-Weight Model Selection and Add Score Tests

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/train.py:155-292`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_train.py`

- [ ] **Step 1: Write failing tests for score calculation**

Add to `tests/test_train.py`:

```python
from dna_sentinel.train import _selection_score


def test_selection_score_equal_weights_by_default():
    metrics = {
        "mobility_balanced_accuracy": 0.6,
        "amr_auroc": 0.9,
        "expansion_auroc": 0.75,
    }
    assert _selection_score(metrics, {"score_mode": "equal"}) == (0.6 + 0.9 + 0.75) / 3


def test_selection_score_legacy_mode_available():
    metrics = {
        "mobility_balanced_accuracy": 0.6,
        "amr_auroc": 0.9,
        "expansion_auroc": 0.75,
    }
    assert _selection_score(metrics, {"score_mode": "legacy"}) == 0.9 + 0.75 + 2.0 * 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_train.py::test_selection_score_equal_weights_by_default tests/test_train.py::test_selection_score_legacy_mode_available -v
```

Expected: FAIL because `_selection_score` does not exist.

- [ ] **Step 3: Implement `_selection_score`**

Add to `src/dna_sentinel/train.py` above `train_cassiopeia`:

```python
def _selection_score(metrics: dict[str, float], config: dict) -> float:
    mob = float(metrics.get("mobility_balanced_accuracy", 0.0))
    amr = float(metrics.get("amr_auroc", 0.0))
    exp = float(metrics.get("expansion_auroc", 0.0))
    mode = config.get("score_mode", "equal")
    if mode == "legacy":
        return amr + exp + 2.0 * mob
    if mode != "equal":
        raise ValueError(f"unknown score_mode: {mode}")
    return (mob + amr + exp) / 3.0
```

Modify training score line:

```python
score = _selection_score(val_metrics, config)
```

Modify print label to show score mode:

```python
f"Mob BA: {val_metrics.get('mobility_balanced_accuracy', 0.0)*100:.1f}% | "
f"Score({config.get('score_mode', 'equal')}): {score:.4f}"
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_train.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dna_sentinel/train.py tests/test_train.py config/cassiopeia_prime.yaml
git commit -m "feat: use equal-weight selection score"
```

---

## Task 5: Add Balanced Sampling Without DataLoader Refactor

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/train.py:172-292`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_train.py`

- [ ] **Step 1: Write failing tests for balanced weights and epoch indices**

Add to `tests/test_train.py`:

```python
import torch

from dna_sentinel.train import _balanced_sample_weights, _epoch_indices


def test_balanced_sample_weights_upweight_rare_labels():
    data = {
        "mobility": torch.tensor([0, 0, 0, 1, 2]),
        "amr": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float),
        "expansion": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float),
    }
    w = _balanced_sample_weights(data)
    assert w.shape == (5,)
    assert torch.isfinite(w).all()
    assert w[4] > w[0]
    assert abs(float(w.mean()) - 1.0) < 1e-6


def test_epoch_indices_respects_balanced_sampling_length():
    data = {
        "mobility": torch.tensor([0, 0, 0, 1, 2]),
        "amr": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float),
        "expansion": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float),
    }
    g = torch.Generator().manual_seed(123)
    idx = _epoch_indices(5, data, {"balanced_sampling": True}, g)
    assert idx.shape == (5,)
    assert idx.min() >= 0
    assert idx.max() < 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_train.py::test_balanced_sample_weights_upweight_rare_labels tests/test_train.py::test_epoch_indices_respects_balanced_sampling_length -v
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement helpers**

Add to `src/dna_sentinel/train.py` above `train_cassiopeia`:

```python
def _inverse_label_frequency(labels: torch.Tensor) -> torch.Tensor:
    y = labels.detach().cpu().long().view(-1)
    counts = torch.bincount(y, minlength=int(y.max().item()) + 1 if y.numel() else 1).float().clamp_min(1.0)
    return 1.0 / counts[y]


def _balanced_sample_weights(data: dict) -> torch.Tensor:
    mob_w = _inverse_label_frequency(data["mobility"])
    amr_w = _inverse_label_frequency(data["amr"].long())
    exp_w = _inverse_label_frequency(data["expansion"].long())
    weights = mob_w + amr_w + exp_w
    return weights / weights.mean().clamp_min(1e-6)


def _epoch_indices(n_train: int, data: dict, config: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    if config.get("balanced_sampling", False):
        weights = _balanced_sample_weights(data)
        return torch.multinomial(weights, n_train, replacement=True, generator=generator)
    return torch.randperm(n_train, generator=generator)
```

Modify training setup:

```python
g = torch.Generator().manual_seed(config.get("seed", 42))
```

Modify epoch index line:

```python
idx = _epoch_indices(n_train, dt, config, g)
```

- [ ] **Step 4: Run tests**

Run:

```bash
python3 -m pytest tests/test_train.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dna_sentinel/train.py tests/test_train.py config/cassiopeia_prime.yaml
git commit -m "feat: add balanced sampling for training"
```

---

## Task 6: Add Deterministic Consistency Feature Cache

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/utils.py:26-34`
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/features.py:178-244`
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/cli.py:20-53`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_fasta.py`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_features.py`

- [ ] **Step 1: Write failing tests for circular shift**

Add to `tests/test_fasta.py`:

```python
from dna_sentinel.utils import circular_shift


def test_circular_shift_wraps_sequence():
    assert circular_shift("AACCGG", 2) == "CCGGAA"
    assert circular_shift("AACCGG", 8) == "CCGGAA"
    assert circular_shift("", 3) == ""
```

- [ ] **Step 2: Write failing test for consistency cache generation**

Add to `tests/test_features.py`:

```python
from dna_sentinel.features import preprocess_consistency_features
from dna_sentinel.utils import LabeledSequence


def test_preprocess_consistency_features_saves_expected_keys(tmp_path):
    records = [
        LabeledSequence("a", "ACGT" * 40, 0, 0, 0),
        LabeledSequence("b", "TGCA" * 40, 1, 1, 0),
    ]
    cfg = CanonicalKmerConfig(window_sizes=(32,), strides=(16,), max_windows=(4,), ngram_min=4, ngram_max=4)
    out = tmp_path / "train_consistency_features.pt"
    preprocess_consistency_features(records, cfg, out, num_workers=1)
    data = torch.load(out, weights_only=True)
    assert set(data) == {"features", "struct_features", "masks", "scale_ids"}
    assert data["features"].shape[0] == 2
    assert data["masks"].shape == (2, 4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_fasta.py::test_circular_shift_wraps_sequence tests/test_features.py::test_preprocess_consistency_features_saves_expected_keys -v
```

Expected: FAIL because `circular_shift` and `preprocess_consistency_features` do not exist.

- [ ] **Step 4: Implement `circular_shift`**

Add to `src/dna_sentinel/utils.py` below `revcomp`:

```python
def circular_shift(seq: str, offset: int) -> str:
    dna = canonical_dna(seq)
    if not dna:
        return dna
    k = offset % len(dna)
    return dna[k:] + dna[:k]
```

- [ ] **Step 5: Implement consistency transform feature generation**

Modify imports in `src/dna_sentinel/features.py`:

```python
from dna_sentinel.utils import LabeledSequence, circular_shift, load_jsonl, revcomp
```

Add below `preprocess_all_features`:

```python
def _consistency_transform(record: LabeledSequence, index: int) -> str:
    if index % 2 == 0:
        return revcomp(record.dna)
    return circular_shift(record.dna, max(1, len(record.dna) // 2))


def preprocess_consistency_features(records: list[LabeledSequence], config: CanonicalKmerConfig,
                                    out_path: str | Path, num_workers: int | None = None) -> None:
    transformed = [LabeledSequence(r.sequence_id, _consistency_transform(r, i), r.mobility, r.amr, r.expansion)
                   for i, r in enumerate(records)]
    preprocess_all_features(transformed, config, out_path, num_workers=num_workers)
```

Modify `extract_features` after saving normal features:

```python
        preprocess_all_features(records, feat_cfg, data_dir / f"{name}_features.pt",
                                 num_workers=kt.get("num_workers", 4))
        if name == "train" and kt.get("build_consistency_cache", False):
            preprocess_consistency_features(records, feat_cfg, data_dir / f"{name}_consistency_features.pt",
                                            num_workers=kt.get("num_workers", 4))
```

- [ ] **Step 6: Load optional consistency cache**

Modify `_load_data` in `src/dna_sentinel/cli.py`:

```python
    cons = data_dir / f"{name}_consistency_features.pt"
    if cons.exists():
        cfeat = torch.load(cons, weights_only=True)
        feat["consistency_features"] = cfeat["features"]
        feat["consistency_struct_features"] = cfeat.get("struct_features")
        feat["consistency_masks"] = cfeat["masks"]
        feat["consistency_scale_ids"] = cfeat.get("scale_ids")
```

Keep this before `return {**feat, **lab}`.

- [ ] **Step 7: Run tests**

Run:

```bash
python3 -m pytest tests/test_fasta.py tests/test_features.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/dna_sentinel/utils.py src/dna_sentinel/features.py src/dna_sentinel/cli.py tests/test_fasta.py tests/test_features.py config/cassiopeia_prime.yaml
git commit -m "feat: add consistency feature cache"
```

---

## Task 7: Add Consistency Regularization to Training

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/train.py:155-292`
- Modify: `/Users/umut/Projeler/bio_spread_project/tests/test_train.py`

- [ ] **Step 1: Write failing tests for consistency loss**

Add to `tests/test_train.py`:

```python
from dna_sentinel.train import _consistency_loss


def test_consistency_loss_is_small_for_identical_outputs():
    out = {
        "mobility_logits": torch.tensor([[2.0, 0.0, -1.0]]),
        "amr_logits": torch.tensor([0.5]),
        "expansion_logits": torch.tensor([-0.25]),
    }
    loss = _consistency_loss(out, out, temperature=1.0)
    assert loss.item() < 1e-6


def test_consistency_loss_positive_for_different_outputs():
    a = {
        "mobility_logits": torch.tensor([[2.0, 0.0, -1.0]]),
        "amr_logits": torch.tensor([0.5]),
        "expansion_logits": torch.tensor([-0.25]),
    }
    b = {
        "mobility_logits": torch.tensor([[-1.0, 0.0, 2.0]]),
        "amr_logits": torch.tensor([-0.5]),
        "expansion_logits": torch.tensor([0.25]),
    }
    loss = _consistency_loss(a, b, temperature=1.0)
    assert loss.item() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_train.py::test_consistency_loss_is_small_for_identical_outputs tests/test_train.py::test_consistency_loss_positive_for_different_outputs -v
```

Expected: FAIL because `_consistency_loss` does not exist.

- [ ] **Step 3: Implement consistency loss**

Add to `src/dna_sentinel/train.py` above `train_cassiopeia`:

```python
def _consistency_loss(out_ref: dict, out_aug: dict, temperature: float = 1.0) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    mob_ref = torch.softmax(out_ref["mobility_logits"].detach() / t, dim=-1)
    mob_aug_log = torch.log_softmax(out_aug["mobility_logits"] / t, dim=-1)
    mob_loss = F.kl_div(mob_aug_log, mob_ref, reduction="batchmean") * (t * t)

    amr_ref = torch.sigmoid(out_ref["amr_logits"].detach())
    amr_aug = torch.sigmoid(out_aug["amr_logits"])
    exp_ref = torch.sigmoid(out_ref["expansion_logits"].detach())
    exp_aug = torch.sigmoid(out_aug["expansion_logits"])
    return mob_loss + F.mse_loss(amr_aug, amr_ref) + F.mse_loss(exp_aug, exp_ref)
```

- [ ] **Step 4: Wire consistency into training**

Inside `train_cassiopeia`, set:

```python
cons_w = float(config.get("consistency_weight", 0.0))
cons_t = float(config.get("consistency_temperature", 1.0))
has_cons = cons_w > 0 and "consistency_features" in dt and exp_cls <= 1
```

Inside the batch after normal tensors:

```python
cons_feat = dt["consistency_features"][bi].to(device) if has_cons else None
cons_mask = dt["consistency_masks"][bi].to(device) if has_cons else None
cons_struct = (dt["consistency_struct_features"][bi].to(device)
               if has_cons and dt.get("consistency_struct_features") is not None else None)
cons_scale = (dt["consistency_scale_ids"][bi].to(device)
              if has_cons and dt.get("consistency_scale_ids") is not None else None)
```

Modify the mixup condition so consistency takes precedence:

```python
if mixup_a > 0 and not has_cons and B > 1 and exp_cls <= 1:
```

After supervised loss and aux loss:

```python
consistency_loss_val = 0.0
if has_cons:
    out_cons = model(cons_feat, cons_mask, struct_features=cons_struct, scale_ids=cons_scale)
    consistency_loss_val = cons_w * _consistency_loss(out, out_cons, cons_t)
loss = _uncertainty_weighted(lm, la, le, model.log_vars) + aux_loss_val + consistency_loss_val
```

- [ ] **Step 5: Run train tests**

Run:

```bash
python3 -m pytest tests/test_train.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dna_sentinel/train.py tests/test_train.py config/cassiopeia_prime.yaml
git commit -m "feat: add consistency regularization"
```

---

## Task 8: Add Compact Benchmark Reporting

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/utils.py`
- Modify: `/Users/umut/Projeler/bio_spread_project/src/dna_sentinel/cli.py`
- Create: `/Users/umut/Projeler/bio_spread_project/tests/test_benchmark.py`

- [ ] **Step 1: Write failing tests for report helpers**

Create `tests/test_benchmark.py`:

```python
import numpy as np

from dna_sentinel.utils import false_positive_summary, task_score


def test_task_score_equal_weight():
    metrics = {
        "mobility_balanced_accuracy": 0.7,
        "amr_auroc": 0.9,
        "expansion_auroc": 0.8,
    }
    assert task_score(metrics) == (0.7 + 0.9 + 0.8) / 3


def test_false_positive_summary_reports_rates_and_quantiles():
    summary = false_positive_summary(
        mobility_probs=np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]]),
        amr_probs=np.array([0.1, 0.8]),
        expansion_probs=np.array([0.2, 0.9]),
        risk_scores=np.array([0.1, 0.8]),
        threshold=0.5,
    )
    assert summary["false_mobile_rate"] == 0.5
    assert summary["false_amr_rate"] == 0.5
    assert summary["false_expansion_rate"] == 0.5
    assert summary["risk_q50"] == 0.45
    assert summary["risk_mean"] == 0.45
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_benchmark.py -v
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement report helpers**

Add to `src/dna_sentinel/utils.py` below metric helpers:

```python
def task_score(metrics: dict[str, float]) -> float:
    return float((metrics.get("mobility_balanced_accuracy", 0.0)
                  + metrics.get("amr_auroc", 0.0)
                  + metrics.get("expansion_auroc", 0.0)) / 3.0)


def false_positive_summary(mobility_probs: np.ndarray, amr_probs: np.ndarray,
                           expansion_probs: np.ndarray, risk_scores: np.ndarray,
                           threshold: float = 0.5) -> dict[str, float]:
    mob = np.asarray(mobility_probs, dtype=float)
    amr = np.asarray(amr_probs, dtype=float)
    exp = np.asarray(expansion_probs, dtype=float)
    risk = np.asarray(risk_scores, dtype=float)
    mobile_prob = 1.0 - mob[:, 0]
    return {
        "false_mobile_rate": float((mobile_prob >= threshold).mean()) if mobile_prob.size else 0.0,
        "false_amr_rate": float((amr >= threshold).mean()) if amr.size else 0.0,
        "false_expansion_rate": float((exp >= threshold).mean()) if exp.size else 0.0,
        "risk_q05": float(np.quantile(risk, 0.05)) if risk.size else 0.0,
        "risk_q50": float(np.quantile(risk, 0.50)) if risk.size else 0.0,
        "risk_q95": float(np.quantile(risk, 0.95)) if risk.size else 0.0,
        "risk_mean": float(risk.mean()) if risk.size else 0.0,
    }
```

- [ ] **Step 4: Add CLI benchmark command**

Add imports in `src/dna_sentinel/cli.py`:

```python
import time
from dna_sentinel.utils import false_positive_summary, task_score
```

Add helper and command below `evaluate_cmd`:

```python
@torch.inference_mode()
def _cached_probabilities(model: Cassiopeia, data: dict, batch_size: int = 64):
    model.eval()
    mob_l, amr_l, exp_l = [], [], []
    sf = data.get("struct_features", None)
    sc = data.get("scale_ids", None)
    n = len(data["features"])
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out = model(data["features"][s:e], data["masks"][s:e],
                    struct_features=sf[s:e] if sf is not None else None,
                    scale_ids=sc[s:e] if sc is not None else None)
        mob_l.append(torch.softmax(out["mobility_logits"], dim=-1).cpu())
        amr_l.append(torch.sigmoid(out["amr_logits"]).cpu())
        if model.config.expansion_classes > 1:
            exp_l.append(torch.softmax(out["expansion_logits"], dim=-1)[:, 1].cpu())
        else:
            exp_l.append(torch.sigmoid(out["expansion_logits"]).cpu())
    mob = torch.cat(mob_l).numpy()
    amr = torch.cat(amr_l).numpy().reshape(-1)
    exp = torch.cat(exp_l).numpy().reshape(-1)
    risk = 0.4 * (1.0 - mob[:, 0]) + 0.3 * amr + 0.3 * exp
    return mob, amr, exp, risk


@cli.command("benchmark")
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--data-dir", default="data/dna_sentinel")
@click.option("--out", default="artifacts/cassiopeia_prime/report.json")
def benchmark_cmd(checkpoint: str, data_dir: str, out: str) -> None:
    model = Cassiopeia.load(checkpoint)
    root = Path(data_dir)
    report = {
        "checkpoint": checkpoint,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_mb": Path(checkpoint).stat().st_size / 1_000_000,
        "config": model.config.to_dict(),
        "splits": {},
    }
    for split in ("val", "test", "heldout_test"):
        if (root / f"{split}_features.pt").exists():
            data = _load_data(root, split, model.config.n_structural_features)
            metrics = evaluate(model, data)
            metrics["task_score"] = task_score(metrics)
            report["splits"][split] = metrics
    if (root / "nonplasmid_control_features.pt").exists():
        data = _load_data(root, "nonplasmid_control", model.config.n_structural_features)
        start = time.perf_counter()
        mob, amr, exp, risk = _cached_probabilities(model, data)
        elapsed = time.perf_counter() - start
        report["splits"]["nonplasmid_control"] = false_positive_summary(mob, amr, exp, risk)
        report["latency_ms_per_cached_sequence"] = 1000.0 * elapsed / max(1, len(data["features"]))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    click.echo(json.dumps(report, indent=2, default=str))
```

This first benchmark command intentionally uses cached features and reports single-class non-plasmid controls as false-positive summaries rather than AUROC. Raw-sequence stress can be added after the core model is trained.

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m pytest tests/test_benchmark.py tests/test_cli_predict.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dna_sentinel/utils.py src/dna_sentinel/cli.py tests/test_benchmark.py
git commit -m "feat: add compact benchmark reporting"
```

---

## Task 9: Full Test, Lint, and Prime Parameter Verification

**Files:**
- No source changes expected unless tests reveal failures.

- [ ] **Step 1: Run full test suite**

Run:

```bash
python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff**

Run:

```bash
python3 -m ruff check src tests
```

Expected: no lint errors.

- [ ] **Step 3: Verify Prime parameter count from config**

Run:

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
cfg = yaml.safe_load(Path('config/cassiopeia_prime.yaml').read_text())
model = Cassiopeia(CassiopeiaConfig.from_yaml(cfg))
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(params)
assert params <= 1_000_000, params
assert params >= 700_000, params
PY
```

Expected: prints a parameter count between 700,000 and 1,000,000.

- [ ] **Step 4: Commit fixes if required**

If files changed during fixes:

```bash
git add src tests config/cassiopeia_prime.yaml
git commit -m "fix: stabilize Prime tests and parameter budget"
```

If no files changed, do not create an empty commit.

---

## Task 10: Train and Benchmark Prime

**Files:**
- Runtime artifacts under `/Users/umut/Projeler/bio_spread_project/artifacts/cassiopeia_prime/`
- Data cache under `/Users/umut/Projeler/bio_spread_project/data/dna_sentinel/train_consistency_features.pt`

- [ ] **Step 1: Build or refresh features with consistency cache**

Run:

```bash
dna-sentinel prepare-features --config config/cassiopeia_prime.yaml
```

Expected:

```text
data/dna_sentinel/train_consistency_features.pt exists
```

Verify:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('data/dna_sentinel/train_consistency_features.pt')
print(p.exists(), p.stat().st_size if p.exists() else 0)
assert p.exists()
PY
```

- [ ] **Step 2: Train Prime**

Run:

```bash
dna-sentinel train --config config/cassiopeia_prime.yaml
```

Expected:

```text
artifacts/cassiopeia_prime/cassiopeia_best.pt exists
artifacts/cassiopeia_prime/cassiopeia_history.json exists
```

- [ ] **Step 3: Benchmark Prime**

Run:

```bash
dna-sentinel benchmark \
  --checkpoint artifacts/cassiopeia_prime/cassiopeia_best.pt \
  --data-dir data/dna_sentinel \
  --out artifacts/cassiopeia_prime/report.json
```

Expected:

```text
artifacts/cassiopeia_prime/report.json exists
```

- [ ] **Step 4: Compare against current checkpoint**

Run:

```bash
dna-sentinel benchmark \
  --checkpoint artifacts/dna_sentinel/cassiopeia_best.pt \
  --data-dir data/dna_sentinel \
  --out artifacts/cassiopeia_prime/baseline_report.json
```

Then run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
base = json.loads(Path('artifacts/cassiopeia_prime/baseline_report.json').read_text())
prime = json.loads(Path('artifacts/cassiopeia_prime/report.json').read_text())
for split in ('val', 'test', 'heldout_test'):
    if split in base['splits'] and split in prime['splits']:
        b = base['splits'][split]['task_score']
        p = prime['splits'][split]['task_score']
        print(split, 'baseline=', round(b, 4), 'prime=', round(p, 4), 'delta=', round(p - b, 4))
PY
```

Expected: Prime should improve at least one of `test` or `heldout_test` task score without a severe drop on the other. Severe drop means more than 0.02 absolute task score loss.

- [ ] **Step 5: Decide next tuning move**

Use this decision table:

| Observation | Next move |
|---|---|
| Prime improves test and heldout | Keep config and document results |
| Prime improves val only | Reduce capacity or dropout; suspect overfit |
| Prime improves drift but hurts AUROC | Lower `consistency_weight` from 0.08 to 0.04 |
| Prime hurts mobility only | Try `window_conv_kernel: 3` |
| Prime hurts AMR/expansion calibration | Reduce `balanced_sampling` effect or disable it for one run |
| Prime exceeds 1M params | Set `frp_out_dim: 320` |

Do not add distillation or attention before running at least two targeted tuning runs from this table.

---

## Task 11: Documentation Only After Metrics Exist

**Files:**
- Modify: `/Users/umut/Projeler/bio_spread_project/README.md`
- Modify: `/Users/umut/Projeler/bio_spread_project/MODEL_CARD.md`

- [ ] **Step 1: Update README only with measured results**

Add a short Prime section after Architecture:

````markdown
## Cassiopeia Prime

Cassiopeia Prime is the focused compact champion configuration for this repository. It keeps inference DNA-only and annotation-free while adding circular plasmid positional encoding, one window-level motif convolution, task-specific evidence windows, balanced sampling, and consistency regularization.

Run:

```bash
dna-sentinel prepare-features --config config/cassiopeia_prime.yaml
dna-sentinel train --config config/cassiopeia_prime.yaml
dna-sentinel benchmark --checkpoint artifacts/cassiopeia_prime/cassiopeia_best.pt --data-dir data/dna_sentinel --out artifacts/cassiopeia_prime/report.json
```
````

Do not write performance claims unless `artifacts/cassiopeia_prime/report.json` exists.

- [ ] **Step 2: Update MODEL_CARD with actual report values**

Add:

```markdown
## Cassiopeia Prime

Prime is a single compact configuration targeting <=1M trainable parameters. It adds CPPE, window motif convolution, task-specific evidence, balanced sampling, and consistency regularization. Metrics below are produced by `dna-sentinel benchmark` and should be regenerated when checkpoints change.
```

Insert actual metrics from `artifacts/cassiopeia_prime/report.json`.

- [ ] **Step 3: Run docs sanity checks**

Run:

```bash
python3 -m pytest tests/test_cli_predict.py tests/test_api_integration.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit docs**

```bash
git add README.md MODEL_CARD.md artifacts/cassiopeia_prime/report.json artifacts/cassiopeia_prime/baseline_report.json
git commit -m "docs: document Cassiopeia Prime results"
```

---

## Final Verification Before Completion

Run:

```bash
python3 -m pytest -q
python3 -m ruff check src tests
python3 - <<'PY'
import yaml
from pathlib import Path
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
cfg = yaml.safe_load(Path('config/cassiopeia_prime.yaml').read_text())
model = Cassiopeia(CassiopeiaConfig.from_yaml(cfg))
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print({'prime_params': params})
assert params <= 1_000_000
PY
```

Completion criteria:

- full tests pass;
- lint passes;
- Prime parameter count <=1M;
- task-specific evidence exists in model output and prediction JSON;
- balanced sampling and consistency are behind config flags;
- benchmark report exists for baseline and Prime;
- README/MODEL_CARD contain only measured claims.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
