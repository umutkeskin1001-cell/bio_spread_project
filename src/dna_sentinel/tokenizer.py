from __future__ import annotations

import torch

from dna_sentinel.fasta import canonical_dna


def window_sequence(seq: str, window: int, stride: int, max_windows: int) -> list[str]:
    if not seq:
        return [""]
    if len(seq) <= window:
        return [seq]
    starts = list(range(0, max(1, len(seq) - window + 1), stride))
    if starts[-1] != len(seq) - window:
        starts.append(len(seq) - window)
    if len(starts) > max_windows:
        if max_windows == 1:
            starts = [starts[len(starts) // 2]]
        else:
            step = (len(starts) - 1) / (max_windows - 1)
            starts = [starts[round(i * step)] for i in range(max_windows)]
    return [seq[s : s + window] for s in starts]


class DnaTokenizer:
    vocab = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

    _ascii_map = torch.zeros(256, dtype=torch.long)
    for _ch, _val in vocab.items():
        _ascii_map[ord(_ch)] = _val
        _ascii_map[ord(_ch.lower())] = _val

    def encode(self, seq: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.zeros(max_len, dtype=torch.long)
        mask = torch.zeros(max_len, dtype=torch.float32)
        length = min(len(seq), max_len)
        if length > 0:
            import numpy as np
            b = bytearray(seq[:length].encode("ascii", errors="ignore"))
            char_tensor = torch.from_numpy(np.frombuffer(b, dtype=np.uint8))
            ids[:length] = self._ascii_map[char_tensor.long()]
            mask[:length] = 1.0
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

