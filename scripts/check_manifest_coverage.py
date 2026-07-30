#!/usr/bin/env python3
"""
CI check: verify that every filename yapsnap can auto-download is listed in
model_checksums.sha256.

Exits 0 if all files are covered, 1 if any are missing. Intended to run in CI
so that a new model added to diarize.py or __init__.py without a manifest
entry is caught before release.

Usage:
    python scripts/check_manifest_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve paths relative to the repo root (one level up from scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yapsnap import load_model_checksums, DEFAULT_MODEL_FILES, LANG_MODELS
from yapsnap.diarize import (
    SEGMENTATION_MODELS,
    EMBEDDING_MODEL_FILE,
)


def main() -> int:
    checksums = load_model_checksums()
    if not checksums:
        print("FAIL: model_checksums.sha256 is missing or empty", file=sys.stderr)
        return 1

    missing: list[str] = []

    # 1) ASR language models: each repo's encoder/decoder/joiner/tokens.
    for lang, repo in sorted(LANG_MODELS.items()):
        for fname in DEFAULT_MODEL_FILES:
            key = f"{repo}/{fname}"
            if key not in checksums:
                missing.append(f"  {key}  (lang={lang})")

    # 2) Diarization segmentation archives (bare filenames in the manifest).
    for seg_key, (archive_name, _, _) in sorted(SEGMENTATION_MODELS.items()):
        if archive_name not in checksums:
            missing.append(f"  {archive_name}  (diarize segmentation, key={seg_key})")

    # 3) Default embedding model.
    if EMBEDDING_MODEL_FILE not in checksums:
        missing.append(f"  {EMBEDDING_MODEL_FILE}  (diarize embedding, default)")

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

    print(
        f"OK: all {len(checksums)} manifest entries verified; "
        f"every downloadable filename is covered.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
