from pathlib import Path

import torch

from dna_sentinel.dataset import DnaDataset, LabeledSequence
from dna_sentinel.model import DnaSentinel, DnaSentinelConfig
from dna_sentinel.train import TrainConfig, evaluate, train_model


def test_training_smoke_produces_checkpoint_and_metrics(tmp_path: Path):
    records = [
        LabeledSequence("p1", "ATGCGT" * 20, 2, 1, 1),
        LabeledSequence("p2", "ATGCGT" * 18, 2, 1, 1),
        LabeledSequence("n1", "TTAACC" * 20, 0, 0, 0),
        LabeledSequence("n2", "TTAACC" * 18, 0, 0, 0),
        LabeledSequence("m1", "CCCCGG" * 20, 1, 0, 0),
        LabeledSequence("m2", "GGCCCC" * 20, 1, 0, 0),
    ]
    ds = DnaDataset(records, window_size=48, stride=24, max_windows=4)
    model = DnaSentinel(DnaSentinelConfig(channels=16, layers=2, window_size=48, stride=24, max_windows=4))
    cfg = TrainConfig(epochs=2, batch_size=3, lr=2e-3, artifact_dir=tmp_path, seed=11)

    ckpt, history = train_model(model, ds, ds, cfg)
    metrics = evaluate(model, ds, batch_size=3)

    assert ckpt.exists()
    assert len(history) == 2
    assert 0.0 <= metrics["amr_auroc"] <= 1.0
    assert torch.load(ckpt, map_location="cpu")["model_config"]["channels"] == 16
