import torch

from dna_sentinel.kmer_transformer import KmerTransformer, KmerTransformerConfig


def test_forward_shapes():
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    out = model(torch.randn(4, 28, 256), torch.ones(4, 28, dtype=torch.bool), torch.zeros(4, 28, dtype=torch.long))
    assert out["mobility_logits"].shape == (4, 3)
    assert out["amr_logits"].shape == (4,)
    assert out["expansion_logits"].shape == (4,)
    assert out["evidence_weights"].shape == (4, 28)
    assert torch.allclose(out["evidence_weights"].sum(dim=1), torch.ones(4), atol=1e-5)

def test_parameter_count():
    cfg = KmerTransformerConfig()
    trainable = sum(p.numel() for p in KmerTransformer(cfg).parameters() if p.requires_grad)
    assert trainable < 120_000

def test_save_load_roundtrip(tmp_path):
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = KmerTransformer.load(path)
    model.eval()
    x = torch.randn(2, 28, 256)
    m = torch.ones(2, 28, dtype=torch.bool)
    s = torch.zeros(2, 28, dtype=torch.long)
    with torch.no_grad():
        a = model(x, m, s)["amr_logits"]
        b = loaded(x, m, s)["amr_logits"]
    assert torch.allclose(a, b)
