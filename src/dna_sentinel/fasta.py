"""FASTA file parsing and sequence normalization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

DNA_ALPHABET = frozenset("ACGT")
_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")

_CANONICAL_TABLE = {i: "N" for i in range(256)}
for char in "ACGT":
    _CANONICAL_TABLE[ord(char)] = char
for char in " \t\r\n-":
    _CANONICAL_TABLE[ord(char)] = None


def canonical_dna(seq: str) -> str:
    upper = seq.upper()
    translated = upper.translate(_CANONICAL_TABLE)
    if any(ord(c) >= 256 for c in translated):
        return "".join(c if c in DNA_ALPHABET else "N" for c in translated)
    return translated


def revcomp(seq: str) -> str:
    return canonical_dna(seq).translate(_RC_TABLE)[::-1]


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    sid: str | None = None
    chunks: list[str] = []
    with Path(path).open("rt", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith(">"):
                if sid is not None:
                    yield sid, canonical_dna("".join(chunks))
                sid = raw[1:].split()[0]
                chunks = []
            else:
                chunks.append(raw)
    if sid is not None:
        yield sid, canonical_dna("".join(chunks))


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path, width: int = 80) -> None:
    with Path(path).open("wt", encoding="utf-8") as handle:
        for sid, seq in records:
            dna = canonical_dna(seq)
            handle.write(f">{sid}\n")
            for i in range(0, len(dna), width):
                handle.write(dna[i : i + width] + "\n")
