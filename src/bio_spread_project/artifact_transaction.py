from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from bio_spread_project.io_utils import ensure_directory, sha256_file, write_json


@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactSet:
    staging_dir: Path
    files: dict[str, Path]


def build_artifact_index(artifact_set: ArtifactSet, final_dir: Path) -> dict[str, object]:
    artifacts: dict[str, ArtifactDescriptor] = {}
    for name, path in artifact_set.files.items():
        target = final_dir / path.name
        artifacts[name] = ArtifactDescriptor(
            name=name,
            relative_path=path.name,
            bytes=target.stat().st_size,
            sha256=sha256_file(target),
        )
    return {
        "schema_version": "artifact_index_v2",
        "artifact_count": len(artifacts),
        "artifacts": {k: v.__dict__ for k, v in artifacts.items()},
    }


def commit_artifact_set(output_dir: Path, artifact_set: ArtifactSet) -> Path:
    ensure_directory(output_dir.parent)
    stage = artifact_set.staging_dir
    if not stage.exists():
        raise ValueError(f"staging directory does not exist: {stage}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    stage.replace(output_dir)
    index = build_artifact_index(
        ArtifactSet(
            staging_dir=output_dir,
            files={name: output_dir / path.name for name, path in artifact_set.files.items()},
        ),
        output_dir,
    )
    write_json(output_dir / "artifact_index.json", index)
    return output_dir
