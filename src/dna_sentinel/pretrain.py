from pathlib import Path

import torch

from dna_sentinel.features import MultiScaleKmerConfig, MultiScaleKmerExtractor
from dna_sentinel.model import KmerTransformer
from dna_sentinel.utils import read_fasta


def iter_pretrain_sequences(
    fasta_path: str | Path,
    limit: int | None = None,
    min_len: int = 0,
    max_len: int | None = None,
):
    seen = 0
    for _, dna in read_fasta(fasta_path):
        if len(dna) < min_len:
            continue
        if max_len is not None and len(dna) > max_len:
            continue
        yield dna
        seen += 1
        if limit is not None and seen >= limit:
            break


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stack_features(
    sequences: list[str],
    extractor: MultiScaleKmerExtractor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, specs, masks, sids = [], [], [], []
    for dna in sequences:
        feat, spec, mask, sid = extractor.extract(dna)
        feats.append(feat)
        specs.append(spec)
        masks.append(mask)
        sids.append(sid)
    return (
        torch.stack(feats).to(device),
        torch.stack(specs).to(device),
        torch.stack(masks).to(device),
        torch.stack(sids).to(device),
    )


def pretrain_mwr(
    model: KmerTransformer,
    fasta_path: str | Path,
    feature_config: MultiScaleKmerConfig,
    config: dict,
) -> list[dict[str, float]]:
    torch.manual_seed(config.get("seed", 42))
    device = _select_device()
    model.to(device)
    extractor = MultiScaleKmerExtractor(feature_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("pretrain_lr", 1e-4),
        weight_decay=config.get("pretrain_weight_decay", config.get("weight_decay", 0.05)),
    )

    epochs = int(config.get("pretrain_epochs", 1))
    batch_size = int(config.get("pretrain_batch_size", config.get("batch_size", 16)))
    mask_ratio = float(config.get("mwr_mask_ratio", 0.15))
    limit = config.get("pretrain_limit")
    min_len = int(config.get("pretrain_min_len", 0))
    max_len = config.get("pretrain_max_len")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        batch: list[str] = []
        for dna in iter_pretrain_sequences(fasta_path, limit=limit, min_len=min_len, max_len=max_len):
            batch.append(dna)
            if len(batch) < batch_size:
                continue
            feat, spec, mask, sid = _stack_features(batch, extractor, device)
            out = model.forward_mwr(feat, spec, mask, sid, mask_ratio=mask_ratio)
            loss = out["mwr_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            batch = []

        if batch:
            feat, spec, mask, sid = _stack_features(batch, extractor, device)
            out = model.forward_mwr(feat, spec, mask, sid, mask_ratio=mask_ratio)
            loss = out["mwr_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        history.append({"epoch": float(epoch), "mwr_loss": avg_loss})
        print(f"Pretrain epoch {epoch:02d}/{epochs:02d} | MWR loss: {avg_loss:.4f}")

    return history
