#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Switch project runtime inputs to SOTA external-source derived datasets")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--archive-legacy", action="store_true", help="Archive existing data/raw files before replacing")
    p.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Refusing to overwrite directory: {dst}")
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def maybe_archive(paths: list[Path], archive_root: Path) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if not p.exists() and not p.is_symlink():
            continue
        target = archive_root / p.name
        if target.exists() or target.is_symlink():
            if target.is_file() or target.is_symlink():
                target.unlink()
        shutil.move(str(p), str(target))


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()

    sota_backbones = root / "data" / "project_inputs" / "silver" / "plasmid_backbones.tsv"
    sota_amr = root / "data" / "project_inputs" / "raw" / "amr.tsv"
    sota_geo = root / "data" / "project_inputs" / "scores" / "backbone_scored.tsv"

    required = [sota_backbones, sota_amr, sota_geo]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing SOTA source files:\n" + "\n".join(missing))

    raw_backbones = root / "data" / "raw" / "plasmid_backbones.tsv"
    raw_amr = root / "data" / "raw" / "amr.tsv"
    raw_geo = root / "data" / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"

    if args.archive_legacy:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_dir = root / "data" / "legacy_archive" / stamp
        maybe_archive([raw_backbones, raw_amr, raw_geo], archive_dir)

    link_or_copy(sota_backbones, raw_backbones, args.copy)
    link_or_copy(sota_amr, raw_amr, args.copy)
    link_or_copy(sota_geo, raw_geo, args.copy)

    print("SOTA source switch complete")
    print(f"- raw_backbones -> {raw_backbones}")
    print(f"- raw_amr -> {raw_amr}")
    print(f"- geo_spread_features -> {raw_geo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
