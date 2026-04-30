from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from bio_spread_project.io_utils import sha256_file

_AST_CACHE: dict[str, tuple[float, str]] = {}


class DocstringStripper(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
        return node


def semantic_table_hash(path: str | Path) -> str:
    path = Path(path)
    if path.suffix == ".parquet":
        lf = pl.scan_parquet(path)
    elif path.suffix in {".tsv", ".tab"}:
        lf = pl.scan_csv(path, separator="\t")
    else:
        lf = pl.scan_csv(path)

    schema = lf.collect_schema()
    numeric_cols = [c for c, t in schema.items() if t.is_numeric()]

    aggs = []
    for c in numeric_cols:
        aggs.extend([
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).mean().alias(f"{c}_mean"),
        ])

    stats = lf.select(aggs).collect().to_dicts()[0] if aggs else {}

    payload = {
        "schema": {c: str(t) for c, t in schema.items()},
        "stats": stats,
    }
    return _hash_payload(payload)


def source_fingerprint(root: Path) -> str:
    src_dir = root / "src" / "bio_spread_project"
    paths = sorted(src_dir.rglob("*.py"))

    fingerprints = {}
    stripper = DocstringStripper()

    for path in paths:
        mtime = os.stat(path).st_mtime
        path_str = str(path)

        if path_str in _AST_CACHE and _AST_CACHE[path_str][0] == mtime:
            fingerprints[path_str] = _AST_CACHE[path_str][1]
            continue

        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        clean_tree = stripper.visit(tree)
        ast_hash = hashlib.sha256(ast.dump(clean_tree).encode("utf-8")).hexdigest()
        _AST_CACHE[path_str] = (mtime, ast_hash)
        fingerprints[path_str] = ast_hash

    return _hash_payload(fingerprints)


def config_fingerprint(root: Path) -> str:
    cfg_root = root / "project_config"
    paths = sorted(
        [
            *cfg_root.rglob("*.json"),
            *cfg_root.rglob("*.yaml"),
            *cfg_root.rglob("*.yml"),
            *cfg_root.rglob("*.txt"),
        ]
    )
    return _hash_files(paths)


def dependency_fingerprint(root: Path) -> str:
    paths = [root / "pyproject.toml", root / "requirements.txt", root / "constraints.txt"]
    return _hash_files([path for path in paths if path.exists()])


def _hash_files(paths: list[Path]) -> str:
    payload = {str(path): sha256_file(path) for path in paths}
    return _hash_payload(payload)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
