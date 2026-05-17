from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from dna_sentinel.fasta import canonical_dna
from dna_sentinel.tokenizer import DnaTokenizer


@dataclass(frozen=True)
class LabeledSequence:
    sequence_id: str
    dna: str
    mobility: int
    amr: int
    expansion: int

    def clean(self) -> "LabeledSequence":
        return LabeledSequence(self.sequence_id, canonical_dna(self.dna), int(self.mobility), int(self.amr), int(self.expansion))


class DnaDataset(Dataset):
    def __init__(self, records: list[LabeledSequence], window_size: int, stride: int, max_windows: int) -> None:
        self.records = [r.clean() for r in records if r.dna]
        self.window_size = window_size
        self.stride = stride
        self.max_windows = max_windows
        self.tokenizer = DnaTokenizer()
        self._tokens: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []
        for rec in self.records:
            tokens, mask = self.tokenizer.batch_windows([rec.dna], self.window_size, self.stride, self.max_windows)
            self._tokens.append(tokens.squeeze(0).to(torch.uint8))
            self._masks.append(mask.squeeze(0).to(torch.bool))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rec = self.records[idx]
        return {
            "sequence_id": rec.sequence_id,
            "tokens": self._tokens[idx].long(),
            "mask": self._masks[idx].float(),
            "mobility": torch.tensor(rec.mobility, dtype=torch.long),
            "amr": torch.tensor(float(rec.amr), dtype=torch.float32),
            "expansion": torch.tensor(float(rec.expansion), dtype=torch.float32),
        }


def save_jsonl(records: list[LabeledSequence], path: str | Path) -> None:
    with Path(path).open("wt", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(asdict(rec.clean()), sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[LabeledSequence]:
    records: list[LabeledSequence] = []
    with Path(path).open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(LabeledSequence(**json.loads(line)))
    return records


def iter_batches(records: list[LabeledSequence], batch_size: int) -> Iterator[list[LabeledSequence]]:
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]
