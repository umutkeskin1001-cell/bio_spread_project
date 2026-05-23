from dna_sentinel.model import KmerTransformer, KmerTransformerConfig
from dna_sentinel.train import _build_optimizer, _score_metrics, _task_loss_multipliers


def test_optimizer_uses_backbone_and_head_learning_rates():
    model = KmerTransformer(KmerTransformerConfig(hidden_dim=16, n_heads=2, n_layers=1, n_kmer_features=128))

    optimizer = _build_optimizer(
        model,
        {
            "lr": 1e-3,
            "backbone_lr": 1e-4,
            "head_lr": 2e-3,
            "weight_decay": 0.05,
        },
    )

    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 2e-3]


def test_score_metrics_accepts_configurable_task_weights():
    metrics = {
        "mobility_balanced_accuracy": 0.7,
        "amr_auroc": 0.8,
        "expansion_auroc": 0.9,
    }

    score = _score_metrics(metrics, {"score_weights": {"mobility": 1.0, "amr": 1.0, "expansion": 2.0}})

    assert score == 3.3


def test_task_loss_multipliers_default_to_one():
    assert _task_loss_multipliers({}) == (1.0, 1.0, 1.0)
    assert _task_loss_multipliers({"expansion_loss_weight": 1.5}) == (1.0, 1.0, 1.5)
