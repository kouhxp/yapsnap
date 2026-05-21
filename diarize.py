"""
yapsnap.diarize — speaker diarization ("who spoke when") on CPU via sherpa-onnx.

This module is self-contained and adds no new runtime dependencies: it reuses
the sherpa-onnx ONNX runtime that yapsnap already requires. It does NOT pull in
PyTorch.

Pipeline (all in the ONNX runtime):

    16kHz mono float32 PCM
        |
        v
    [segmentation]  pyannote-3.0 (default) or reverb-v1 (--diarize-model reverb)
        |                neural speaker-activity map -> turn boundaries
        v
    [embedding]     3D-Speaker CAM++ (en/zh) -> per-segment voiceprints
        |
        v
    [FastClustering] group voiceprints into speaker IDs
        |
        v
    list[SpeakerTurn]  (start, end, speaker) in ORIGINAL-time seconds

Model licenses (the sherpa-onnx code itself is Apache-2.0):
    - pyannote-segmentation-3.0 : CC-BY-4.0  (default; attribution only)
    - reverb-diarization-v1     : NON-COMMERCIAL  (opt-in via --diarize-model reverb)
    - 3D-Speaker CAM++ embedding: Apache-2.0
"""

from __future__ import annotations

import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
#
# Segmentation models ship as .tar.bz2 archives that extract to a directory
# containing model.onnx. Embedding models ship as a bare .onnx.
#
# All URLs are GitHub release assets from k2-fsa/sherpa-onnx, which the
# existing _download() helper can fetch (it only special-cases tokens.txt vs
# .onnx for sanity checks; we extend that below for .tar.bz2).

_SEG_BASE = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-segmentation-models")
_EMB_BASE = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models")  # NB: upstream spells it "recongition"

# segmentation key -> (archive filename, extracted dir name, license note)
SEGMENTATION_MODELS = {
    "pyannote": (
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0",
        "CC-BY-4.0 (attribution)",
    ),
    "reverb": (
        "sherpa-onnx-reverb-diarization-v1.tar.bz2",
        "sherpa-onnx-reverb-diarization-v1",
        "NON-COMMERCIAL — see Rev.ai model card before any commercial use",
    ),
}

DEFAULT_SEGMENTATION = "pyannote"

import os

# The en/zh CAM++ model the user asked about. Multilingual, ~27MB, Apache-2.0.
# Overridable via env var in case the exact release-asset name differs:
#   export YAPSNAP_EMBEDDING_MODEL=3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
# Known-good alternatives in the speaker-recongition-models release:
#   3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
#   3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
EMBEDDING_MODEL_FILE = os.environ.get(
    "YAPSNAP_EMBEDDING_MODEL",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
)


@dataclass
class SpeakerTurn:
    """A contiguous span attributed to one speaker, in original-time seconds."""
    start: float
    end: float
    speaker: int  # 0-based speaker index

    @property
    def label(self) -> str:
        return f"SPEAKER_{self.speaker:02d}"


# ---------------------------------------------------------------------------
# Model resolution / download
# ---------------------------------------------------------------------------

def _extract_tar_bz2(archive: Path, dest_dir: Path) -> None:
    """Extract a .tar.bz2 into dest_dir (its parent), guarding against path
    traversal in member names."""
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tf:
        base = dest_dir.parent.resolve()
        for member in tf.getmembers():
            target = (base / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tf.extractall(dest_dir.parent)


def ensure_diarization_models(
    seg_key: str,
    cache_dir: Path,
    download_fn,
) -> tuple[Path, Path]:
    """Ensure the segmentation + embedding models exist in cache_dir.

    `download_fn(url, dest)` is yapsnap's existing _download helper. We pass it
    in rather than importing to keep this module independent of yapsnap import
    order and easy to unit-test.

    Returns (segmentation_model_onnx, embedding_model_onnx).
    """
    if seg_key not in SEGMENTATION_MODELS:
        raise ValueError(
            f"unknown segmentation model {seg_key!r}; "
            f"choose from {sorted(SEGMENTATION_MODELS)}"
        )

    archive_name, extract_dir_name, license_note = SEGMENTATION_MODELS[seg_key]
    cache_dir.mkdir(parents=True, exist_ok=True)

    seg_dir = cache_dir / extract_dir_name
    seg_onnx = seg_dir / "model.onnx"
    if not seg_onnx.is_file():
        if seg_key == "reverb":
            print(
                "note: reverb-diarization-v1 is NON-COMMERCIAL licensed. "
                "Verify the Rev.ai model card permits your use case.",
                file=sys.stderr,
            )
        archive_path = cache_dir / archive_name
        if not archive_path.is_file():
            download_fn(f"{_SEG_BASE}/{archive_name}", archive_path)
        print(f"  extracting {archive_name}", file=sys.stderr)
        _extract_tar_bz2(archive_path, seg_dir)
        archive_path.unlink(missing_ok=True)
        if not seg_onnx.is_file():
            raise RuntimeError(
                f"after extracting {archive_name}, {seg_onnx} is missing"
            )

    emb_onnx = cache_dir / EMBEDDING_MODEL_FILE
    if not emb_onnx.is_file():
        download_fn(f"{_EMB_BASE}/{EMBEDDING_MODEL_FILE}", emb_onnx)

    return seg_onnx, emb_onnx


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def _build_diarizer(seg_onnx: Path, emb_onnx: Path, num_speakers: int,
                    cluster_threshold: float, num_threads: int):
    """Construct an OfflineSpeakerDiarization, guarding for older sherpa-onnx
    builds that predate the diarization API."""
    import sherpa_onnx

    missing = [
        name for name in (
            "OfflineSpeakerDiarization",
            "OfflineSpeakerDiarizationConfig",
            "OfflineSpeakerSegmentationModelConfig",
            "OfflineSpeakerSegmentationPyannoteModelConfig",
            "SpeakerEmbeddingExtractorConfig",
            "FastClusteringConfig",
        )
        if not hasattr(sherpa_onnx, name)
    ]
    if missing:
        ver = getattr(sherpa_onnx, "__version__", "unknown")
        raise RuntimeError(
            f"this sherpa-onnx build ({ver}) lacks the speaker-diarization API "
            f"(missing: {', '.join(missing)}). Upgrade with: "
            f"pip install -U 'sherpa-onnx>=1.10'"
        )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_onnx)
            ),
            num_threads=num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_onnx),
            num_threads=num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,        # -1 => auto
            threshold=cluster_threshold,      # only used when num_clusters == -1
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "sherpa-onnx rejected the diarization config; check that the "
            "segmentation and embedding model files exist and are valid."
        )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def diarize_pcm(
    samples,
    seg_onnx: Path,
    emb_onnx: Path,
    *,
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
    num_threads: int = 1,
    show_progress: bool = False,
) -> list[SpeakerTurn]:
    """Run diarization on 16kHz mono float32 PCM (numpy array).

    `samples` MUST be decoded at original speed (speed=1.0). Diarization on
    sped-up audio degrades both segmentation boundaries and embedding quality.

    Returns turns sorted by start time, in original-time seconds.
    """
    import numpy as np

    if samples is None or len(samples) == 0:
        return []
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    sd = _build_diarizer(
        seg_onnx, emb_onnx, num_speakers, cluster_threshold, num_threads
    )

    if show_progress:
        def _cb(done: int, total: int) -> int:
            if total:
                print(f"\r  diarizing: {100.0 * done / total:.0f}%",
                      end="", file=sys.stderr)
            return 0
        result = sd.process(samples, callback=_cb).sort_by_start_time()
        print("", file=sys.stderr)
    else:
        result = sd.process(samples).sort_by_start_time()

    return [SpeakerTurn(start=float(r.start), end=float(r.end),
                        speaker=int(r.speaker)) for r in result]


# ---------------------------------------------------------------------------
# Assigning speakers to timestamped transcript sentences
# ---------------------------------------------------------------------------

def speaker_at(turns: list[SpeakerTurn], t: float) -> Optional[int]:
    """Speaker index whose turn contains time t, or the nearest turn's speaker
    if t falls in a gap. Returns None only when there are no turns."""
    if not turns:
        return None
    best = None
    best_gap = None
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
        # distance to this turn if t is outside it
        gap = turn.start - t if t < turn.start else t - turn.end
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = turn.speaker
    return best


def label_sentences(
    sentences: list[tuple[float, str]],
    turns: list[SpeakerTurn],
) -> list[tuple[int | None, float, str]]:
    """Attach a speaker index to each (start_time, text) sentence.

    Input sentences carry original-time start seconds (yapsnap already scales
    by `speed`), so they share a clock with `turns`.
    """
    return [(speaker_at(turns, t), t, text) for (t, text) in sentences]
