#!/usr/bin/env python3
"""
CI check: verify that every filename yapsnap can auto-download is listed in
model_checksums.sha256.

Exits 0 if all files are covered, 1 if any are missing. Intended to run in CI
so that a new model added to diarize.py or __init__.py without a manifest
entry is caught before release.

Zero runtime dependencies — parses the source files directly instead of
importing yapsnap (which pulls in numpy, sherpa-onnx, etc.).

Usage:
    python scripts/check_manifest_coverage.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "yapsnap" / "model_checksums.sha256"
INIT_PY = REPO_ROOT / "yapsnap" / "__init__.py"
DIARIZE_PY = REPO_ROOT / "yapsnap" / "diarize.py"


def parse_manifest() -> dict[str, str]:
    """Parse model_checksums.sha256 into {name: digest}."""
    checksums: dict[str, str] = {}
    if not MANIFEST.is_file():
        return checksums
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            checksums[parts[1].strip()] = parts[0].strip()
    return checksums


def extract_dict_from_source(path: Path, var_name: str) -> dict:
    """Extract a top-level dict assignment from a Python source file via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"{var_name} not found in {path}")


def extract_tuple_from_source(path: Path, var_name: str) -> tuple:
    """Extract a top-level tuple assignment from a Python source file via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"{var_name} not found in {path}")


def extract_embedding_default(path: Path) -> str:
    """Extract the default EMBEDDING_MODEL_FILE value from diarize.py.

    It's set via os.environ.get(..., <default>), so we grab the second arg
    from the source with a regex (AST won't help since it's a call, not a
    literal assignment).
    """
    src = path.read_text(encoding="utf-8")
    m = re.search(
        r'EMBEDDING_MODEL_FILE\s*=\s*os\.environ\.get\(\s*'
        r'["\']YAPSNAP_EMBEDDING_MODEL["\']\s*,\s*["\']([^"\']+)["\']\s*',
        src,
    )
    if not m:
        raise ValueError("could not parse EMBEDDING_MODEL_FILE default from diarize.py")
    return m.group(1)


def main() -> int:
    checksums = parse_manifest()
    if not checksums:
        print("FAIL: model_checksums.sha256 is missing or empty", file=sys.stderr)
        return 1

    # From __init__.py
    lang_models = extract_dict_from_source(INIT_PY, "LANG_MODELS")
    model_files = extract_tuple_from_source(INIT_PY, "DEFAULT_MODEL_FILES")

    # From diarize.py
    seg_models = extract_dict_from_source(DIARIZE_PY, "SEGMENTATION_MODELS")
    emb_default = extract_embedding_default(DIARIZE_PY)

    missing: list[str] = []

    # 1) ASR language models
    for lang, repo in sorted(lang_models.items()):
        for fname in model_files:
            key = f"{repo}/{fname}"
            if key not in checksums:
                missing.append(f"  {key}  (lang={lang})")

    # 2) Diarization segmentation archives
    for seg_key, (archive_name, _, _) in sorted(seg_models.items()):
        if archive_name not in checksums:
            missing.append(f"  {archive_name}  (diarize segmentation, key={seg_key})")

    # 3) Default embedding model
    if emb_default not in checksums:
        missing.append(f"  {emb_default}  (diarize embedding, default)")

    if missing:
        print(
            "FAIL: the following downloadable files are missing from "
            "model_checksums.sha256:\n" + "\n".join(missing),
            file=sys.stderr,
        )
        print(
            "\nRun scripts/gen_hashes.sh --models to regenerate the manifest.",
            file=sys.stderr,
        )
        return 1

    n_checked = len(lang_models) * len(model_files) + len(seg_models) + 1
    print(f"OK: all {n_checked} downloadable filenames found in manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
