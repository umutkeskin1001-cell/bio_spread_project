from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

DNA_ALPHABET = frozenset("ACGT")
_RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def canonical_dna(seq: str) -> str:
    """Return uppercase A/C/G/T/N DNA; gaps and whitespace are removed."""
    out: list[str] = []
    for ch in seq.upper():
        if ch in {" ", "\t", "\r", "\n", "-"}:
            continue
        out.append(ch if ch in DNA_ALPHABET else "N")
    return "".join(out)


def revcomp(seq: str) -> str:
    return canonical_dna(seq).translate(_RC_TABLE)[::-1]


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    sid: str | None = None
    chunks: list[str] = []
    with Path(path).open("rt", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    yield sid, canonical_dna("".join(chunks))
                sid = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if sid is not None:
        yield sid, canonical_dna("".join(chunks))


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path, width: int = 80) -> None:
    with Path(path).open("wt", encoding="utf-8") as handle:
        for sid, seq in records:
            dna = canonical_dna(seq)
            handle.write(f">{sid}\n")
            for i in range(0, len(dna), width):
                handle.write(dna[i : i + width] + "\n")
