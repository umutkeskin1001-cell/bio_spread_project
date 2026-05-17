from __future__ import annotations

import torch

from dna_sentinel.fasta import canonical_dna


def window_sequence(seq: str, window: int, stride: int, max_windows: int) -> list[str]:
    dna = canonical_dna(seq)
    if not dna:
        return [""]
    if len(dna) <= window:
        return [dna]
    starts = list(range(0, max(1, len(dna) - window + 1), stride))
    if starts[-1] != len(dna) - window:
        starts.append(len(dna) - window)
    if len(starts) > max_windows:
        if max_windows == 1:
            starts = [starts[len(starts) // 2]]
        else:
            step = (len(starts) - 1) / (max_windows - 1)
            starts = [starts[round(i * step)] for i in range(max_windows)]
    return [dna[s : s + window] for s in starts]


class DnaTokenizer:
    vocab = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

    def encode(self, seq: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        dna = canonical_dna(seq)[:max_len]
        ids = torch.zeros(max_len, dtype=torch.long)
        mask = torch.zeros(max_len, dtype=torch.float32)
        for i, ch in enumerate(dna):
            ids[i] = self.vocab.get(ch, 0)
            mask[i] = 1.0
        return ids, mask

    def batch_windows(
        self,
        sequences: list[str],
        window_size: int,
        stride: int,
        max_windows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.zeros(len(sequences), max_windows, window_size, dtype=torch.long)
        masks = torch.zeros(len(sequences), max_windows, window_size, dtype=torch.float32)
        for i, seq in enumerate(sequences):
            windows = window_sequence(seq, window_size, stride, max_windows)
            for j, win in enumerate(windows[:max_windows]):
                ids, mask = self.encode(win, window_size)
                tokens[i, j] = ids
                masks[i, j] = mask
        return tokens, masks
