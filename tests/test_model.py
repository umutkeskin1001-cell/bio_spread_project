import torch

from dna_sentinel.model import KmerTransformer, KmerTransformerConfig


def test_model_forward_shapes_are_task_specific():
    cfg = KmerTransformerConfig(hidden_dim=16, n_heads=2, n_layers=1, n_kmer_features=128)
    model = KmerTransformer(cfg)

    kmer_feat = torch.randn(2, 28, 128)
    spec_feat = torch.randn(2, 28, 512)
    mask = torch.ones(2, 28, dtype=torch.bool)
    scale_ids = torch.zeros(2, 28, dtype=torch.long)

    out = model(kmer_feat, spec_feat, mask, scale_ids)

    assert out["mobility_logits"].shape == (2, 3)
    assert out["amr_logits"].shape == (2,)
    assert out["expansion_logits"].shape == (2,)
    assert out["evidence_weights"].shape == (2, 28)
    assert torch.allclose(out["evidence_weights"].sum(dim=1), torch.ones(2), atol=1e-5)
