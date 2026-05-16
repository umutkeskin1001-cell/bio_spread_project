# BioSpread İyileştirme Planı — Tam Kapsamlı Kılavuz

> Hedef: Temporal AUC'yi 0.90+ seviyesine çıkarmak, cold-start AUC'yi 0.68 → 0.80+ seviyesine taşımak, 18 örnek sorununu kökünden çözmek.

---

## İçindekiler
1. [Derin Kök Neden Analizi](#1-derin-kök-neden-analizi)
2. [Faz 1: Veri ve Altyapı (Hafta 1)](#2-faz-1-veri-ve-altyapı)
3. [Faz 2: Mimari İyileştirmeler (Hafta 2)](#3-faz-2-mimari-i̇yileştirmeler)
4. [Faz 3: Eğitim ve Soğuk-Start (Hafta 3)](#4-faz-3-eğitim-ve-soğuk-start)
5. [Faz 4: İleri Seviye (Hafta 4)](#5-faz-4-i̇leri-seviye)
6. [Faz 5: Ölçümleme ve İzleme](#6-faz-5-ölçümleme-ve-i̇zleme)
7. [Uygulama Yol Haritası](#7-uygulama-yol-haritası)
8. [Başarı Kriterleri](#8-başarı-kriterleri)

---

## 1. Derin Kök Neden Analizi

### 1.1 Cold-Start Neden Başarısız (ROC AUC ~0.68)?

```
KÖK NEDEN AĞACI:

Cold-Start AUC 0.68
├── Test seti çok küçük (18 örnek) → güvenilir ölçüm yok
├── Cold-start head mimarisi çok zayıf
│   ├── Sadece 1 layer MLP (128→64→3)
│   ├── Lambda=0.25 ile çok düşük loss ağırlığı
│   └── Taksonomi embedding'ini direkt görmüyor
├── Temporal masking yetersiz
│   ├── Sadece %30 batch maskeleniyor
│   ├── "All or nothing" — kısa history backbone'ları yeterince zorlanmıyor
│   └── Null embedding öğrenilebilir ama sabit — adaptif değil
├── Static encoder zayıf
│   └── Basit MLP (gated highway kaldırılmış)
├── Distribution shift
│   └── Test backbone'larının positive rate'i train'den farklı (%47 vs %36)
└── Hiçbir cold-start-specifik optimizasyon yok
    ├── Hard negative mining yok
    ├── Contrastive learning yok
    ├── Distillation yok
    └── Prototypical networks yok
```

### 1.2 Temporal Neden Orta Seviye (ROC AUC ~0.86-0.90)?

```
KÖK NEDEN AĞACI:

Temporal AUC 0.86-0.90
├── Veri Sorunları
│   ├── n_hosts ve niche_breadth her zaman 0 → gürültü
│   ├── 31% backbone tek yıllık → temporal encoder bunlar için işe yaramaz
│   └── Dead feature'lar dimension'ı şişiriyor
├── Loss Fonksiyonu Sorunları
│   ├── Count head lambda=0.15 → neredeyse hiç öğrenmiyor
│   ├── Ranking loss lambda=0.10 → çok düşük
│   ├── Loss scaling sadece 50 batch → stabil değil
│   └── El ile ayarlanmış lambdalar → optimal değil
├── Temporal Encoder Sorunları
│   ├── GRU 2 layer → yetersiz kapasite
│   ├── Attention sadece softmax → inductive bias zayıf
│   └── Feature cross-attention yok
├── Calibration Sorunları
│   └── Platt scaler sadece 1/7 val seti ile → az veri
└── Evaluation Sorunları
    └── F2 threshold val'de seçiliyor → overfit riski
```

---

## 2. FAZ 1: Veri ve Altyapı (Hafta 1)

### Değişiklik 1A: Synthetic Cold-Start Evaluation

**Hedef:** 18 örnek sorununu hemen bypass et

**Yapılacak değişiklikler:**
- `src/bio_spread/cli/main.py` — `evaluate` komutuna yeni bir `--synthetic-cold` flag'i ekle
- Yeni fonksiyon: `evaluate_synthetic_cold_start(model, val_loader, device)`

**Uygulama:**
```python
# cli/main.py içinde evaluate fonksiyonuna ekle
@click.option("--synthetic-cold/--no-synthetic-cold", default=False)
def evaluate(model_path, config, feature_dir, output_path, synthetic_cold):
    # ... mevcut kod ...
    if synthetic_cold:
        # Val backbonelarını temporal mask'le
        val_bids = split.get("val", [])
        val_df = seq_df.filter(...)
        # temporal_mask = True ile forward pass
        # Cold Platt scaler'ı kullan (torch.no_grad)
        # Tüm metrikleri hesapla
```

**Değiştirilecek dosyalar:**
- `src/bio_spread/cli/main.py` — evaluate komutuna parametre ekle
- `src/bio_spread/models/trainer.py` — yeni `evaluate_cold` metodu

**Success metric:** 942 val backbone'u üzerinden hesaplanan synthetic cold-start ROC AUC

**Risk:** Düşük — sadece var olan temporal_mask=True mantığını eval'de kullanıyor

---

### Değişiklik 1B: Dead Feature Cleanup

**Hedef:** Gürültüyü azalt, dimension'ı küçült, modelin öğrenmesini kolaylaştır

**Yapılacak değişiklikler:**

`src/bio_spread/constants.py`:
```python
# MEVCUT:
STATIC_FEATURES = [
    "log_size", "gc", "n_replicon_types", "n_relaxase_types",
    "mobility_score", "is_conjugative", "is_mobilizable",
    "topology", "n_orit_types", "host_range_rank",
]

# YENİ:
STATIC_FEATURES = [
    "log_size", "gc",
    "n_replicon_types", "has_relaxase", "n_relaxase_types",
    "mobility_score", "is_conjugative", "is_mobilizable",
    "topology", "has_orit",  # has_orit = (n_orit_types > 0)
    "host_range_rank",
]
# n_hosts ve niche_breadth → KALDIRILDI (always 0)
```

`src/bio_spread/data/snapshot.py` — `build_sequences` içinde:
```python
# Yeni feature'ları oluştur
snapshots = snapshots.with_columns([
    (pl.col("n_orit_types") > 0).cast(pl.Float64).alias("has_orit"),
    (pl.col("n_relaxase_types") > 0).cast(pl.Float64).alias("has_relaxase"),
])
# n_hosts ve niche_breadth'i düşür
snapshots = snapshots.drop(["n_hosts", "niche_breadth"])
```

**Değiştirilecek dosyalar:**
- `src/bio_spread/constants.py` — feature listeleri
- `src/bio_spread/data/snapshot.py` — feature engineering
- `src/bio_spread/data/dataset.py` — feature dimension güncelle
- `tests/` — feature sayısı değiştiği için güncelle

**Success metric:** Static feature dimension 10 → 11 (ama her biri anlamlı), snapshot feature dimension 16 → 14 (dead feature'lar çıktı)

**Risk:** Düşük. Sadece her zaman 0 olan feature'ları kaldırıyoruz.

---

### Değişiklik 1C: Per-Backbone Temporal Split

**Hedef:** Gerçek temporal split ile test setini büyüt, her backbone'un geleceğini tahmin et

**Uygulama:**

`src/bio_spread/data/snapshot.py` — yeni split fonksiyonu:
```python
def temporal_backbone_split(
    df: pl.DataFrame,
    split_year: int = 2022,
    val_frac: float = 0.15,
) -> tuple[set[str], set[str], set[str]]:
    """
    Her backbone için: cutoff_year öncesi train, sonrası test.
    cutoff_year = split_year - spread_horizon (yani future data için yer bırak)
    
    Örnek: split_year=2022, spread_horizon=3
    - cutoff = 2019
    - backbone'un 2019 öncesi gözlemleri → train
    - backbone'un 2020 sonrası gözlemleri → test
    - Her backbone HEM train HEM test verisi üretir
    """
    all_bids = df["backbone_id"].unique().to_list()
    np.random.shuffle(all_bids)
    n_val = max(1, int(len(all_bids) * val_frac))
    val_bids = set(all_bids[:n_val])
    remaining = set(all_bids[n_val:])
    
    cutoff = split_year - 3  # 3 yıl future için yer bırak
    
    # Her backbone için temporal split
    train_bids = set()  # cutoff öncesi gözlemi olan backbonelar
    test_bids = set()   # cutoff sonrası gözlemi olan backbonelar
    
    for bid in remaining:
        years = df.filter(pl.col("backbone_id") == bid)["year"].unique().to_list()
        has_past = any(y <= cutoff for y in years)
        has_future = any(y > cutoff for y in years)
        if has_past:
            train_bids.add(bid)
        if has_future:
            test_bids.add(bid)
    
    return train_bids, val_bids, test_bids
```

`src/bio_spread/data/snapshot.py` — `build_sequences` içinde per-backbone temporal split:
```python
# Her snapshot'a split ata (backbone bazlı değil, snapshot bazlı!)
sequences = sequences.with_columns(
    pl.when(pl.col("year") <= cutoff)
    .then(pl.lit("train"))
    .otherwise(pl.lit("test"))
    .alias("split")
)
```

**Değiştirilecek dosyalar:**
- `src/bio_spread/data/snapshot.py` — split mantığı
- `src/bio_spread/cli/main.py` — prepare komutu
- `src/bio_spread/data/dataset.py` — observed filtresi

**Expected impact:** Test seti 18'den **binlerce** örneğe çıkar. Temporal ROC AUC düşebilir (çünkü test artık daha zor — backbone'u daha önce gördüğüm halde geleceğini tahmin etmeliyim) ama daha gerçekçi olur.

**Risk:** Orta. Eğer bir backbone'un train ve test gözlemleri çok benzer yıllardaysa (mesela 2019 train, 2020 test), temporal leakage olabilir. `spread_horizon` kadar boşluk bırakarak çözülür.

---

### Değişiklik 1D: Feature Engineering

`src/bio_spread/data/snapshot.py` — yeni feature'lar:

```python
def build_sequences(...):
    # --- Mevcut feature'lar ---
    # ...
    
    # --- Yeni feature'lar ---
    # 1. AMR gen zenginliği (eğer AMR verisi varsa)
    if "amr_gene_count" in df.columns:
        snapshots = snapshots.with_columns([
            pl.col("amr_gene_count").log1p().alias("log_amr_genes"),
        ])
    
    # 2. Ülke embedding'i (ilk 10 ülke one-hot, gerisi "other")
    top_countries = ["USA", "China", "UK", "Germany", "France", ...]
    for c in top_countries:
        snapshots = snapshots.with_columns(
            (pl.col("country") == c).cast(pl.Float64).alias(f"country_{c}")
        )
    
    # 3. Yayılma hızının ivmesi (mevcut acceleration'a ek)
    # acceleration zaten var: new_countries_recent - new_countries_2y_ago
    # Buna ek olarak: 4 yıllık trend
    snapshots = snapshots.with_columns([
        (pl.col("new_countries_recent") / (pl.col("years_since_first") + 1).clip(1))
        .alias("spread_velocity_norm"),
    ])
    
    # 4. Host diversity (gerçek host verisi varsa)
    # Şu an n_hosts=0, eğer doldurulursa:
    if "n_hosts" in df.columns and df["n_hosts"].max() > 0:
        snapshots = snapshots.with_columns([
            (pl.col("n_hosts") / (pl.col("n_countries") + 1)).alias("host_per_country"),
        ])
```

**Değiştirilecek dosyalar:**
- `src/bio_spread/data/snapshot.py`
- `src/bio_spread/constants.py`
- `src/bio_spread/data/dataset.py`

**Expected impact:** Feature sayısı artar, ama her biri anlamlı. AMR verisi entegre edilebilirse etkisi büyük olur.

**Risk:** Düşük-Orta. AMR verisi yoksa skip edilir, diğer feature'lar zaten var olan veriden türetilir.

---

## 3. FAZ 2: Mimari İyileştirmeler (Hafta 2)

### Değişiklik 2A: Residual Gated StaticEncoder

**Hedef:** Static feature'ların daha iyi temsil edilmesi

`src/bio_spread/models/components.py` — yeni sınıf:
```python
class GatedResidualMLP(nn.Module):
    """Highway-inspired gated residual MLP.
    
    output = gate(x) * main(x) + (1 - gate(x)) * skip(x)
    
    Bu sayede:
    - Gradient highway: derin ağlarda bile gradient kaybolmaz
    - Adaptive gating: model hangi transform'u uygulayacağını öğrenir
    - Residual connections: daha stabil eğitim
    """
    def __init__(self, dims: list[int], dropout: float = 0.15):
        super().__init__()
        assert len(dims) >= 2
        self.main = MLP(dims, dropout=dropout)
        in_dim, out_dim = dims[0], dims[-1]
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        main_out = self.main(x)
        skip_out = self.skip(x)
        return gate * main_out + (1 - gate) * skip_out
```

`src/bio_spread/models/sovereign.py` — güncelle:
```python
class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, static_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = GatedResidualMLP(
            [input_dim, static_dim * 2, static_dim * 2, static_dim],
            dropout=dropout,
        )  # 3 layer → daha derin
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
```

Ayrıca `ColdStartHead`'e aynı gated yapıyı ekle:
```python
class ColdStartHead(nn.Module):
    def __init__(self, static_dim: int, tax_embed_dim: int, n_hazard: int = 3, dropout: float = 0.1):
        super().__init__()
        # Cold-start head direkt taksonomi embedding'ini de alsın
        input_dim = static_dim + tax_embed_dim  # 128 + 40 = 168
        self.net = GatedResidualMLP(
            [input_dim, input_dim // 2, n_hazard],
            dropout=dropout,
        )
    
    def forward(self, z_static: torch.Tensor, tax_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_static, tax_embed], dim=-1)
        return self.net(x)
```

**Değiştirilecek dosyalar:**
- `src/bio_spread/models/components.py` — yeni GatedResidualMLP
- `src/bio_spread/models/sovereign.py` — StaticEncoder, ColdStartHead güncelle
- `src/bio_spread/models/__init__.py` — export

**Expected impact:** Static encoding kalitesi artar. Gradient highway sayesinde daha derin ağ kullanılabilir.

---

### Değişiklik 2B: Dual Cold-Start Architecture

**Hedef:** Cold-start için ayrı, güçlü bir prediction path'i oluştur

**Yeni mimari:**
```
forward() çağrısı:
  static (B, 10) + taxonomy_idxs (B, 5)
      |
  TaxonomyEncoder → tax_embed (B, 40)
      |
  Concat → static_input (B, 50)
      |
  GatedStaticEncoder
      |
  +----+----+
  |         |
z_static  z_static
  |         |
  |    [ColdStartHead]
  |    GatedMLP(168→84→3)  ← 168 = 128 + 40 (taxonomy direkt!)
  |         |
  |    cold_logits (B, 3)
  |
  TemporalEncoder (+ masking)
  |
  FusionGate → fused (B, 128)
  |
  HazardHead → hazard_logits (B, 3)
```

`src/bio_spread/models/sovereign.py`:
```python
def forward(self, static, snapshots, mask, taxonomy_idxs=None, temporal_mask=None):
    if self.use_taxonomy and taxonomy_idxs is not None:
        tax_emb, _ = self.taxonomy_encoder(taxonomy_idxs)
        static_input = torch.cat([static, tax_emb], dim=-1)
    else:
        tax_emb = None
        static_input = static
    
    z_static = self.static_encoder(static_input)
    
    # Cold-start head: taksonomi embedding'ini DIREKT al
    if self.use_taxonomy and tax_emb is not None:
        cold_logits = self.cold_start_head(z_static, tax_emb)
    else:
        cold_logits = self.cold_start_head(z_static, torch.zeros_like(z_static))
    
    # Temporal path (mevcut)
    h_all, h_pooled = self.temporal_encoder(snapshots, mask, temporal_mask)
    h_pooled_proj = self.temporal_proj(h_pooled)
    fused, gate_weights = self.gate(z_static, h_pooled_proj)
    h = self.hazard_proj(fused)
    hazard_logits = self.hazard_head(h)
    # ...
```

---

### Değişiklik 2C: Gradual Temporal Masking

**Hedef:** Her backbone'un history uzunluğuna göre adaptif masking

`src/bio_spread/models/trainer.py`:
```python
def _get_adaptive_temporal_mask(
    self, seq_lens: torch.Tensor, epoch: int, max_epochs: int
) -> torch.Tensor:
    """
    History'si kısa olan backbone'lar DAHA FAZLA maskelenir.
    Epoch ilerledikçe masking oranı artar (curriculum).
    """
    B = seq_lens.size(0)
    
    # Base masking probability: epoch'a göre artar
    progress = epoch / max_epochs
    base_prob = 0.2 + 0.4 * progress  # 20% → 60%
    
    # History uzunluğuna göre adjust et
    # 1 yıl history → 2x base_prob
    # 10+ yıl history → 0.5x base_prob
    history_weight = torch.clamp(1.0 / (seq_lens.float().sqrt()), min=0.5, max=2.0)
    
    probs = base_prob * history_weight
    probs = torch.clamp(probs, min=0.05, max=0.9)
    
    mask = torch.rand(B, device=probs.device) < probs
    return mask
```

`src/bio_spread/models/trainer.py` — `_train_epoch` içinde:
```python
# MEVCUT:
temporal_mask = torch.rand(B, device=self.device) < self.temporal_masking_prob

# YENİ:
temporal_mask = self._get_adaptive_temporal_mask(
    batch["seq_len"], epoch, self.epochs
)
```

---

### Değişiklik 2D: Cross-Attention Fusion

**Hedef:** Temporal bilginin hangi kısmının önemli olduğunu öğren

`src/bio_spread/models/sovereign.py` — `BioSpreadModel.__init__`:
```python
# FusionGate'e ek olarak cross-attention
if self.use_cross_attention:
    self.cross_attn_q = nn.Linear(static_dim, static_dim)
    self.cross_attn_k = nn.Linear(temporal_dim, static_dim)
    self.cross_attn_v = nn.Linear(temporal_dim, static_dim)
    self.cross_attn_out = nn.Linear(static_dim, static_dim)
```

`forward`:
```python
# Cross-attention: static → temporal
if self.use_cross_attention:
    q = self.cross_attn_q(z_static).unsqueeze(1)  # (B, 1, 128)
    k = self.cross_attn_k(h_all_proj)              # (B, L, 128)
    v = self.cross_attn_v(h_all_proj)              # (B, L, 128)
    
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(128)  # (B, 1, L)
    attn = attn.masked_fill(~mask.unsqueeze(1), -1e9)
    attn = F.softmax(attn, dim=-1)                # (B, 1, L)
    
    context = torch.matmul(attn, v).squeeze(1)    # (B, 128)
    
    # Fusion gate'e ek girdi
    gate_input = torch.cat([z_static, h_pooled_proj, context], dim=-1)  # (B, 384)
else:
    gate_input = torch.cat([z_static, h_pooled_proj], dim=-1)
```

---

## 4. FAZ 3: Eğitim ve Soğuk-Start (Hafta 3)

### Değişiklik 3A: Uncertainty-Weighted Multi-Task Loss

`src/bio_spread/models/trainer.py`:
```python
class AdaptiveLossWeighting(nn.Module):
    """Kendiliğinden ayarlanan loss ağırlıkları.
    
    Her loss component'i için bir sigma öğrenilir.
    total_loss = sum_i (loss_i / (2 * sigma_i^2) + log(sigma_i))
    """
    def __init__(self, n_losses: int = 6):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(n_losses))
    
    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, (name, loss) in enumerate(losses.items()):
            sigma = torch.exp(self.log_sigmas[i])
            total += loss / (2 * sigma) + 0.5 * self.log_sigmas[i]  # log(sigma)
        return total
```

**Trainer'da kullanım:**
```python
self.loss_weighting = AdaptiveLossWeighting(n_losses=6)
# optimizer'a ekle:
optimizer.add_param_group({"params": self.loss_weighting.parameters(), "lr": 1e-3})
```

---

### Değişiklik 3B: Curriculum Learning

`src/bio_spread/models/trainer.py` — `_train_epoch`:
```python
def _get_curriculum_params(self, epoch: int):
    """Epoch'a göre training parametrelerini ayarla."""
    progress = epoch / self.epochs
    
    params = {}
    
    if epoch <= 3:
        # Phase 1: Sadece temporal (warmup)
        params["temporal_masking_prob"] = 0.0
        params["lambda_cold"] = 0.0
        params["lambda_rank"] = 0.0
        params["noise_std"] = 0.0
    elif epoch <= 8:
        # Phase 2: Hafif masking + cold-start
        params["temporal_masking_prob"] = 0.1 + 0.2 * ((epoch - 3) / 5)
        params["lambda_cold"] = 0.1 + 0.15 * ((epoch - 3) / 5)
        params["lambda_rank"] = 0.05
        params["noise_std"] = 0.02
    else:
        # Phase 3: Full training
        params["temporal_masking_prob"] = 0.3 + 0.3 * min(progress, 1.0)
        params["lambda_cold"] = 0.5
        params["lambda_rank"] = 0.15
        params["noise_std"] = 0.05 + 0.03 * (progress - 0.16) / 0.84
    
    return params
```

---

### Değişiklik 3C: Hard Negative Mining for Cold-Start

`src/bio_spread/models/trainer.py`:
```python
def _cold_start_hard_negative_loss(
    self, out: ModelOutput, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Cold-start head'in en çok yanıldığı örneklere odaklan.
    """
    # Cold-start logits
    cold_logits = out.cold_logits
    cold_probs = torch.sigmoid(cold_logits)
    
    # Her örnek için loss
    per_sample_loss = F.binary_cross_entropy_with_logits(
        cold_logits, targets, reduction="none"
    ).mean(dim=-1)  # (B,) — 3 horizon ortalaması
    
    # En yüksek loss'a sahip %20'ye 2x ağırlık ver
    k = max(1, len(per_sample_loss) // 5)
    hard_indices = torch.topk(per_sample_loss, k).indices
    
    weights = torch.ones_like(per_sample_loss)
    weights[hard_indices] = 2.0
    
    # Ayrıca: false negative'leri ekstra cezalandır
    # (model "yayılmaz" dedi ama yayıldı)
    fn_mask = (cold_probs < 0.5) & (targets == 1.0)
    fn_penalty = fn_mask.float().sum(dim=-1) / 3.0  # (B,) 0-1 arası
    weights = weights + fn_penalty * 0.5
    
    return (per_sample_loss * weights).mean()
```

---

### Değişiklik 3D: Knowledge Distillation

**Amaç:** Temporal modelin bilgisini static-only modele aktar

`scripts/train_distill.py` — yeni script:
```python
"""
1. Teacher modeli normal temporal data ile eğit
2. Student modeli (sadece static) teacher'ın temporal-masked çıktılarını taklit et
"""

def distill_loss(
    student_logits: torch.Tensor,      # student'ın cold_logits
    teacher_logits_masked: torch.Tensor, # teacher'ın temporal-masked hazard_logits
    targets: torch.Tensor,               # gerçek etiketler
    temperature: float = 3.0,
    alpha: float = 0.7,                  # distillation weight
) -> torch.Tensor:
    """
    Loss = alpha * KL(student || teacher) + (1-alpha) * BCE(student, targets)
    """
    # Soft targets from teacher (with temperature)
    teacher_probs = F.softmax(teacher_logits_masked / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(student_log_soft, teacher_probs, reduction="batchmean")
    kl_loss = kl_loss * (temperature ** 2)  # Scale back
    
    # Hard target BCE
    bce_loss = F.binary_cross_entropy_with_logits(student_logits, targets)
    
    return alpha * kl_loss + (1 - alpha) * bce_loss
```

**Eğitim prosedürü:**
```
Adım 1: Teacher modelini normal şekilde eğit (mevcut kod)
Adım 2: Student modeli oluştur (static-only, temporal encoder YOK)
Adım 3: Student'ı eğit:
  - Teacher'dan temporal-masked logits al
  - Student'tan cold_start_head çıktısı al
  - Distillation loss ile student'ı optimize et
Adım 4: Student modelini cold-start inference'da kullan
```

**Değiştirilecek dosyalar:**
- `scripts/train_distill.py` — yeni script
- `src/bio_spread/models/sovereign.py` — student model opsiyonel

**Expected impact:** Cold-start AUC'de en büyük sıçrama buradan gelir. Temporal modelin zengin bilgisi static modele distillemiş olur.

---

### Değişiklik 3E: Prototypical Networks

`src/bio_spread/models/prototypical.py` — yeni dosya:
```python
class PrototypicalColdStart(nn.Module):
    """
    Prototypical Networks ile cold-start tahmini.
    
    Embedding uzayında her backbone bir nokta.
    Cold-start backbone'u, en yakın train backbone'larının
    spread pattern'ine göre sınıflandırılır.
    """
    def __init__(self, backbone_embeddings: torch.Tensor, labels: torch.Tensor, n_horizons: int = 3):
        """
        Args:
            backbone_embeddings: (N_train, embed_dim) — train backbone embedding'leri
            labels: (N_train, n_horizons) — train backbone etiketleri
        """
        super().__init__()
        self.register_buffer("embeddings", backbone_embeddings)
        self.register_buffer("labels", labels)
        self.k = 10  # Number of nearest neighbors
    
    def predict(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """
        KNN + weighted average.
        
        Args:
            query_embedding: (B, embed_dim)
        Returns:
            (B, n_horizons) — cold-start probability
        """
        # Cosine similarity
        sim = F.normalize(query_embedding, dim=-1) @ F.normalize(self.embeddings, dim=-1).T
        # (B, N_train)
        
        # Top-K neighbors
        weights, indices = torch.topk(sim, self.k, dim=-1)  # (B, K)
        weights = F.softmax(weights / 0.1, dim=-1)  # Temperature
        
        # Weighted average of labels
        neighbor_labels = self.labels[indices]  # (B, K, 3)
        pred = (weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
        return pred
```

**Kullanımı:**
```python
# Eğitim sonrası:
train_embeddings = model.get_embedding(train_static, train_seq, train_mask, train_tax)
train_labels = train_hazard_final  # (N, 3)

prototypical = PrototypicalColdStart(train_embeddings, train_labels)

# Cold-start inference:
test_embedding = model.get_embedding(test_static, test_seq, test_mask, test_tax, temporal_mask=True)
cold_pred = prototypical.predict(test_embedding)

# Final prediction: model output + prototypical weighted average
final_pred = 0.6 * model_cold_logits + 0.4 * prototypical_pred
```

---

## 5. FAZ 4: İleri Seviye (Hafta 4)

### Değişiklik 4A: Self-Supervised Pre-Training

`scripts/pretrain_masked.py` — yeni script:
```python
"""
Masked Snapshot Modeling ile self-supervised pre-training.

Hedef: Temporal dinamikleri anlamak için modeli ön eğit.
Sonra hazard prediction için fine-tune et.
"""

def pretrain_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        seq = batch["seq"].to(device)          # (B, L, 16)
        mask = batch["mask"].to(device)        # (B, L)
        
        # Snapshot'ların %15'ini maskala
        B, L, F = seq.shape
        masking_prob = torch.full((B, L), 0.15, device=device)
        masking_prob = masking_prob * mask     # Sadece valid timestep'ler
        masking_mask = torch.rand(B, L, device=device) < masking_prob
        
        # Maskalanan feature'ları sıfırla
        seq_masked = seq.clone()
        seq_masked[masking_mask] = 0.0
        
        # Model'den maskalanan yerleri tahmin et
        with torch.no_grad():
            static = batch["static"].to(device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(device)
        
        # Forward pass (model'in temporal encoder'ını kullan)
        z_static = model.static_encoder(
            torch.cat([static, model.taxonomy_encoder(taxonomy_idxs)[0]], dim=-1)
            if model.use_taxonomy else static
        )
        h_all, _ = model.temporal_encoder(seq_masked, mask)
        
        # Maskalanan feature'ları geri inşa et (regression head)
        pred_features = model.pretrain_head(h_all)  # (B, L, 16)
        
        # Loss: sadece maskalanan yerlerde MSE
        reconstruction_loss = F.mse_loss(
            pred_features[masking_mask],
            seq[masking_mask],
            reduction="mean",
        )
        
        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()
        
        total_loss += reconstruction_loss.item()
    
    return total_loss / len(loader)
```

**Pre-training sonrası fine-tune:**
```
1. Pre-train: 50 epoch masked reconstruction
2. Fine-tune: mevcut hazard prediction loss ile 50 epoch
3. Beklenen: temporal encoder daha iyi initialized olduğu için daha hızlı converge ve daha iyi AUC
```

---

### Değişiklik 4B: Contrastive Learning

`src/bio_spread/models/contrastive.py` — yeni dosya:
```python
class TemporalContrastiveHead(nn.Module):
    """
    SimCLR-style contrastive learning for backbone embeddings.
    
    Positive pair: Aynı backbone'un farklı zaman dilimleri
    Negative pair: Farklı backbone'lar
    """
    def __init__(self, embed_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(z), dim=-1)


def contrastive_loss(
    z1: torch.Tensor,  # (B, proj_dim) — view 1
    z2: torch.Tensor,  # (B, proj_dim) — view 2 (same backbone, different time)
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    NT-Xent loss.
    - z1[i] ve z2[i] → positive pair
    - z1[i] ve z2[j≠i] → negative pair
    """
    B = z1.size(0)
    
    # Cosine similarity matrix
    sim = z1 @ z2.T / temperature  # (B, B)
    
    # Labels: diagonal = positive
    labels = torch.arange(B, device=z1.device)
    
    loss = F.cross_entropy(sim, labels)  # each row: predict which j matches i
    
    return loss
```

---

### Değişiklik 4C: Calibration

`src/bio_spread/models/trainer.py` — mevcut Platt scaler'ı güçlendir:

```python
def _learn_platt(self, loader, scalers, force_temporal_mask=False):
    """
    Mevcut Platt scaling'i aynı bırak, ancak calibration verisini artır.
    
    MEVCUT: val setinin 1/7'si (~88 backbone)
    YENİ: val setinin TAMAMI (~619 backbone)
    """
    # ... mevcut kod ...
    # Değişiklik: calibration split'ini kaldır
    # calibration verisi olarak val setinin tamamını kullan
```

Ayrıca opsiyonel Beta calibration ekle:
```python
class BetaCalibrator(nn.Module):
    """Beta calibration: Platt'dan daha esnek.
    P(y=1|f) = 1 / (1 + exp(-a * log(f/(1-f)) - b))
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        log_odds = torch.log(p / (1 - p))
        return self.a * log_odds + self.b
```

---

## 6. FAZ 5: Ölçümleme ve İzleme

### Değişiklik 5A: Bootstrap Confidence Intervals

`src/bio_spread/utils/metrics.py`:
```python
def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict:
    """
    Her metrik için bootstrap güven aralığı hesapla.
    
    Örnek:
        bootstrap_metrics(y_true, y_pred, roc_auc_score)
        → {"mean": 0.73, "ci_low": 0.62, "ci_high": 0.84}
    """
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_low": float(np.percentile(scores, alpha * 100)),
        "ci_high": float(np.percentile(scores, (1 - alpha) * 100)),
    }
```

### Değişiklik 5B: Feature Importance (SHAP)

`scripts/feature_importance.py` — yeni script:
```python
"""
Integrated Gradients ile her tahmin için feature importance.
"""
def compute_feature_importance(model, sample, device, n_steps=50):
    model.eval()
    static = sample["static"].to(device).requires_grad_(True)
    seq = sample["seq"].to(device)
    mask = sample["mask"].to(device)
    
    # Baseline: sıfır static feature'lar
    baseline = torch.zeros_like(static)
    
    # Integrated gradients
    scaled_inputs = [
        baseline + (float(i) / n_steps) * (static - baseline)
        for i in range(n_steps + 1)
    ]
    
    grads = []
    for inp in scaled_inputs:
        out = model(inp, seq, mask)
        logits = out.hazard_logits
        # Gradient of output w.r.t input
        grad = torch.autograd.grad(logits.sum(), inp)[0]
        grads.append(grad)
    
    # Trapezoidal integration
    grads = torch.stack(grads)
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0
    integrated_grads = (static - baseline) * avg_grads
    
    return integrated_grads.detach().cpu().numpy()
```

---

## 7. Uygulama Yol Haritası

```
Hafta 1: Veri ve Altyapı (5 iş günü)
├── Gün 1-2:   Synthetic cold-start eval (1A)
│               → Hemen kullanılabilir, 18 örnek sorununu bypass eder
│               → Çıktı: --synthetic-cold flag'i ile çalışan evaluation
│
├── Gün 3:     Dead feature cleanup (1B)
│               → n_hosts, niche_breadth kaldır, has_orit/has_relaxase ekle
│               → Test'leri güncelle
│               → Çıktı: Temiz feature set, güncel test'ler
│
└── Gün 4-5:   Per-backbone temporal split (1C)
│               → Snapshot-bazlı split
│               → Test seti 18 → binlerce örnek
│               → Çıktı: Yeni split mantığı, büyük test seti

Hafta 2: Mimari İyileştirmeler (5 iş günü)
├── Gün 1-2:   GatedResidualMLP + Dual cold-start head (2A, 2B)
│               → Static encoder'ı highway'e çevir
│               → Cold-start head taksonomi alsın
│               → Çıktı: Yeni StaticEncoder, ColdStartHead
│
├── Gün 3:     Gradual temporal masking (2C)
│               → History uzunluğuna göre adaptif masking
│               → Curriculum learning ile masking oranını artır
│               → Çıktı: _get_adaptive_temporal_mask()
│
└── Gün 4-5:   Cross-attention fusion (2D)
│               → Static→Temporal cross attention
│               → Fusion gate'te ek girdi
│               → Çıktı: Cross-attention modülü

Hafta 3: Eğitim ve Soğuk-Start (5 iş günü)
├── Gün 1:     Uncertainty-weighted loss (3A)
│               → AdaptiveLossWeighting sınıfı
│               → Çıktı: Otomatik loss ağırlıklandırma
│
├── Gün 2:     Curriculum learning (3B) + Hard negative mining (3C)
│               → 3 fazlı eğitim
│               → Cold-start hard negative loss
│               → Çıktı: _get_curriculum_params(), _cold_start_hard_negative_loss()
│
├── Gün 3-4:   Knowledge distillation (3D)
│               → Teacher model eğit
│               → Student model eğit
│               → Çıktı: scripts/train_distill.py
│
└── Gün 5:     Prototypical networks (3E)
│               → KNN-based cold-start
│               → Ensemble with model output
│               → Çıktı: PrototypicalColdStart sınıfı

Hafta 4: İleri Seviye + Kalibrasyon (5 iş günü)
├── Gün 1-2:   Self-supervised pre-training (4A)
│               → Masked snapshot modeling
│               → scripts/pretrain_masked.py
│               → Çıktı: Pre-train script, fine-tune pipeline
│
├── Gün 3:     Contrastive learning (4B)
│               → Temporal contrastive head
│               → Çıktı: TemporalContrastiveHead, contrastive_loss()
│
├── Gün 4:     Calibration (4C) + Bootstrap (5A)
│               → Tam val seti ile Platt scaling
│               → Bootstrap CI
│               → Çıktı: Daha iyi kalibrasyon, güven aralıkları
│
└── Gün 5:     Feature importance (5B) + Final entegrasyon
│               → SHAP/Integrated Gradients
│               → Tüm değişiklikleri CLI/serving'e entegre et
│               → Çıktı: scripts/feature_importance.py
```

---

## 8. Başarı Kriterleri

### Hedef Metrikler

| Metrik | Şu An | Hedef | Nasıl Ölçülür |
|--------|-------|-------|--------------|
| Temporal ROC AUC (h3) | 0.860 | **0.920+** | Validation set |
| Temporal F1 | 0.759 | **0.820+** | Validation set |
| Cold-start ROC AUC (h3) | 0.688 | **0.800+** | Synthetic cold-start eval |
| Cold-start Recall | 0.273 | **0.500+** | Synthetic cold-start eval |
| ECE (calibration error) | 0.053 | **<0.030** | Validation set |
| Brier score | 0.155 | **<0.120** | Validation set |
| Test set size | 18 | **1000+** | Per-backbone temporal split |
| Bootstrapped CI width | N/A | **<0.05** | Bootstrap 1000 iterations |

### Minimum Kabul Kriterleri

1. Temporal ROC AUC ≥ 0.900 (validation)
2. Cold-start ROC AUC ≥ 0.750 (synthetic eval)
3. Test seti ≥ 500 örnek
4. ECE < 0.050
5. Tüm mevcut test'ler geçiyor (regression)
6. CLI ve serving API çalışıyor

### İzleme Metrikleri

Her eğitim sonunda otomatik rapor:
```
=== FAZ 3 SONU ===
Temporal:
  ROC AUC: 0.912 [0.901-0.923]
  F1: 0.803
  Precision: 0.798 | Recall: 0.808
  ECE: 0.028 | Brier: 0.112

Cold-Start (synthetic, n=3,247):
  ROC AUC: 0.782 [0.768-0.796]
  F1: 0.615
  Precision: 0.712 | Recall: 0.541

Cold-Start (gerçek test, n=18):
  ROC AUC: 0.751 [0.680-0.822]
  F1: 0.523

Test set: 3,452 samples (per-backbone temporal split)
```

---

## EK: Değişiklik Özeti Tablosu

| # | Değişiklik | Dosyalar | Etki | Zorluk | Öncelik |
|---|---|---|---|---|---|
| 1A | Synthetic cold-start eval | `cli/main.py`, `trainer.py` | ★★★★★ | Kolay | 🔴 Acil |
| 1B | Dead feature cleanup | `constants.py`, `snapshot.py`, `dataset.py` | ★★★ | Kolay | 🔴 Acil |
| 1C | Per-backbone temporal split | `snapshot.py`, `cli/main.py` | ★★★★★ | Orta | 🔴 Acil |
| 2A | GatedResidualMLP | `components.py`, `sovereign.py` | ★★★★ | Kolay | 🟡 Önemli |
| 2B | Dual cold-start head | `sovereign.py`, `trainer.py` | ★★★★★ | Kolay | 🟡 Önemli |
| 2C | Gradual temporal masking | `trainer.py` | ★★★★ | Orta | 🟡 Önemli |
| 2D | Cross-attention | `sovereign.py` | ★★★★ | Zor | 🟢 İyileştirme |
| 3A | Uncertainty-weighted loss | `trainer.py` | ★★★★ | Orta | 🟡 Önemli |
| 3B | Curriculum learning | `trainer.py` | ★★★ | Orta | 🟢 İyileştirme |
| 3C | Hard negative mining | `trainer.py` | ★★★★ | Orta | 🟡 Önemli |
| 3D | Knowledge distillation | `train_distill.py`, `sovereign.py` | ★★★★★ | Zor | 🟡 Önemli |
| 3E | Prototypical networks | `prototypical.py` | ★★★★ | Orta | 🟢 İyileştirme |
| 4A | Self-supervised pre-training | `pretrain_masked.py` | ★★★★★ | Zor | 🟢 İyileştirme |
| 4B | Contrastive learning | `contrastive.py`, `trainer.py` | ★★★★ | Zor | 🟢 İyileştirme |
| 4C | Better calibration | `trainer.py`, `components.py` | ★★★ | Kolay | 🟡 Önemli |
| 5A | Bootstrap CI | `metrics.py` | ★★★ | Kolay | 🟡 Önemli |
| 5B | Feature importance | `feature_importance.py` | ★★★ | Orta | 🟢 İyileştirme |

---

> **Not:** Bu plan modülerdir. Her adım bağımsız olarak uygulanabilir ve test edilebilir.
> Önerilen başlangıç: **Faz 1'in tamamı** + **Faz 2'nin 2A, 2B, 2C maddeleri** — bunlar en yüksek etki/zorluk oranına sahip.
