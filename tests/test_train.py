from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.train import _build_optimizer


def test_optimizer_uses_backbone_and_head_lr():
    model = Cassiopeia(CassiopeiaConfig(
        n_canonical_features=128, hidden_dim=16))
    opt = _build_optimizer(model, {"lr": 1e-3, "backbone_lr": 1e-4, "head_lr": 2e-3, "weight_decay": 0.05})
    groups = opt.param_groups
    assert len(groups) >= 2
    assert groups[0]["lr"] == 1e-4
    assert groups[-1]["lr"] == 2e-3
