import torch

from dna_sentinel.model import KmerTransformer, KmerTransformerConfig


def test_forward_shapes():
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    out = model(torch.randn(4, 28, 256), torch.randn(4, 28, 512), torch.ones(4, 28, dtype=torch.bool), torch.zeros(4, 28, dtype=torch.long))
    assert out["mobility_logits"].shape == (4, 3)
    assert out["amr_logits"].shape == (4,)
    assert out["expansion_logits"].shape == (4,)
    assert out["evidence_weights"].shape == (4, 28)
    assert torch.allclose(out["evidence_weights"].sum(dim=1), torch.ones(4), atol=1e-5)


def test_parameter_count():
    # Legacy parameter count check
    cfg_legacy = KmerTransformerConfig(task_specific_pooling=False, scale_isolated_conv=False)
    trainable_legacy = sum(p.numel() for p in KmerTransformer(cfg_legacy).parameters() if p.requires_grad)
    assert trainable_legacy == 436_809
    assert trainable_legacy < 500_000

    # New model parameter count check
    cfg_new = KmerTransformerConfig(task_specific_pooling=True, scale_isolated_conv=True)
    trainable_new = sum(p.numel() for p in KmerTransformer(cfg_new).parameters() if p.requires_grad)
    assert trainable_new == 436_980
    assert trainable_new < 500_000


def test_forward_mwr_reconstructs_masked_windows():
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    kmer = torch.randn(2, 28, 256)
    spec = torch.randn(2, 28, 512)
    mask = torch.ones(2, 28, dtype=torch.bool)
    scale_ids = torch.zeros(2, 28, dtype=torch.long)
    mwr_mask = torch.zeros(2, 28, dtype=torch.bool)
    mwr_mask[:, :3] = True

    out = model.forward_mwr(kmer, spec, mask, scale_ids, mwr_mask=mwr_mask)

    assert out["reconstruction"].shape == (2, 28, 32)
    assert out["target"].shape == (2, 28, 32)
    assert torch.equal(out["mwr_mask"], mwr_mask)
    assert out["mwr_loss"].ndim == 0
    assert out["mwr_loss"].item() >= 0.0


def test_load_accepts_pre_upgrade_checkpoints_missing_new_modules(tmp_path):
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    state = {
        "config": cfg.to_dict(),
        "state_dict": {
            name: value
            for name, value in model.state_dict().items()
            if not name.startswith("local_conv.") and not name.startswith("recon_head.")
        },
    }
    path = tmp_path / "legacy.pt"
    torch.save(state, path)

    loaded = KmerTransformer.load(path)

    out = loaded(
        torch.randn(2, 28, 256),
        torch.randn(2, 28, 512),
        torch.ones(2, 28, dtype=torch.bool),
        torch.zeros(2, 28, dtype=torch.long),
    )
    assert out["mobility_logits"].shape == (2, 3)


def test_load_rejects_checkpoints_missing_core_weights(tmp_path):
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    state_dict = model.state_dict()
    del state_dict["lex_proj.0.weight"]
    path = tmp_path / "broken.pt"
    torch.save({"config": cfg.to_dict(), "state_dict": state_dict}, path)

    try:
        KmerTransformer.load(path)
    except RuntimeError as exc:
        assert "missing checkpoint keys" in str(exc)
    else:
        raise AssertionError("Expected broken checkpoint to fail loading")


def test_save_load_roundtrip(tmp_path):
    cfg = KmerTransformerConfig(hidden_dim=32, n_heads=2, n_layers=1, n_kmer_features=256)
    model = KmerTransformer(cfg)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = KmerTransformer.load(path)
    model.eval()
    x = torch.randn(2, 28, 256)
    sp = torch.randn(2, 28, 512)
    m = torch.ones(2, 28, dtype=torch.bool)
    s = torch.zeros(2, 28, dtype=torch.long)
    with torch.no_grad():
        a = model(x, sp, m, s)["amr_logits"]
        b = loaded(x, sp, m, s)["amr_logits"]
    assert torch.allclose(a, b)
