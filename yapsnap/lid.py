"""
yapsnap.lid — spoken-language identification for ``--lang auto``.

Like diarize.py, this module adds no runtime dependencies: it drives
sherpa-onnx's SpokenLanguageIdentification API, the same ONNX runtime yapsnap
already uses for ASR. No PyTorch, no onnxruntime Python package.

The model is a whisper-tiny export pruned to yapsnap's ten languages
(https://huggingface.co/kouhxp/whisper-tiny-lid): the decoder's token embedding
keeps only the <|sot|> row and the output projection only the ten language rows,
with the encoder's metadata rewritten so sherpa-onnx maps logit index i to
LANGS[i]. Everything sherpa-onnx needs — mel settings, language codes — travels
inside the encoder's metadata_props, so there is nothing to parse here.

Pipeline:

    two 5-second probe windows, 16kHz mono float32 at speed=1.0
        |
        v
    [sherpa-onnx SpokenLanguageIdentification]  whisper encoder + one decoder
        |                                        step at the SOT position
        v
    a two-letter code per window -> agreement, or a tiebreak on both together

Detection is advisory: the caller wraps everything here and falls back to
English rather than failing a transcription run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

LID_REPO = "kouhxp/whisper-tiny-lid"

# The two files sherpa-onnx needs. Language codes and mel settings live in the
# encoder's metadata_props, so no side-car json/npz is required.
LID_ENCODER_FILE = "tiny-encoder.int8.onnx"
LID_DECODER_FILE = "tiny-decoder.int8.onnx"
LID_FILES = (LID_ENCODER_FILE, LID_DECODER_FILE)

# A local directory holding both files above bypasses auto-download (useful
# offline, or to pin your own export).
LID_MODEL_DIR_ENV = "YAPSNAP_LID_MODEL_DIR"

SAMPLE_RATE = 16000

# Probe window length. Whisper pads anything shorter than 30s internally, so a
# few 5s windows identify the language in a fraction of a second on one core —
# far cheaper than feeding it the whole recording. yapsnap takes two of them;
# see PROBE_POSITIONS in __init__.py for where they are cut from.
WINDOW_SECONDS = 5.0

# Whisper emits ISO-639-1 codes. The pruned model is built with yapsnap's own
# spellings baked into the encoder metadata, so this only matters if someone
# points YAPSNAP_LID_MODEL_DIR at a stock whisper export.
WHISPER_CODE_ALIASES = {
    "he": "iw",   # Hebrew: whisper says "he", yapsnap's repo key is "iw"
    "jw": "jv",
    "in": "id",
}


# ---------------------------------------------------------------------------
# Model download / resolution
# ---------------------------------------------------------------------------

def local_model_dir() -> Optional[Path]:
    """Directory from YAPSNAP_LID_MODEL_DIR, or None when unset."""
    raw = os.environ.get(LID_MODEL_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def ensure_lid_model(cache_dir: Path, download_fn) -> tuple[Path, Path]:
    """Ensure the LID encoder + decoder exist, downloading what's missing.

    `download_fn(url, dest, repo=...)` is yapsnap's `_download` helper, passed in
    rather than imported so this module stays independent of import order (the
    same convention as diarize.ensure_diarization_models). Every downloaded file
    goes through the SHA-256 manifest check inside `_download`.

    Returns (encoder_path, decoder_path).
    """
    override = local_model_dir()
    if override is not None:
        missing = [f for f in LID_FILES if not (override / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{LID_MODEL_DIR_ENV}={override} is missing: {', '.join(missing)}"
            )
        return override / LID_ENCODER_FILE, override / LID_DECODER_FILE

    cache_dir.mkdir(parents=True, exist_ok=True)
    needed = [f for f in LID_FILES if not (cache_dir / f).is_file()]
    if needed:
        print(f"Downloading language-ID model from {LID_REPO} to {cache_dir}",
              file=sys.stderr)
    for fname in needed:
        url = f"https://huggingface.co/{LID_REPO}/resolve/main/{fname}"
        download_fn(url, cache_dir / fname, repo=LID_REPO)
    return cache_dir / LID_ENCODER_FILE, cache_dir / LID_DECODER_FILE


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def _build_slid(encoder: Path, decoder: Path, num_threads: int):
    """Construct a SpokenLanguageIdentification, guarding for older sherpa-onnx
    builds that predate the API."""
    import sherpa_onnx

    missing = [
        name for name in (
            "SpokenLanguageIdentification",
            "SpokenLanguageIdentificationConfig",
            "SpokenLanguageIdentificationWhisperConfig",
        )
        if not hasattr(sherpa_onnx, name)
    ]
    if missing:
        ver = getattr(sherpa_onnx, "__version__", "unknown")
        raise RuntimeError(
            f"this sherpa-onnx build ({ver}) lacks the spoken-language-ID API "
            f"(missing: {', '.join(missing)}). Upgrade with: "
            f"pip install -U 'sherpa-onnx>=1.13'"
        )

    config = sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=str(encoder),
            decoder=str(decoder),
        ),
        num_threads=max(1, num_threads),
        provider="cpu",
        debug=False,
    )
    return sherpa_onnx.SpokenLanguageIdentification(config)


def _compute(slid, samples: np.ndarray) -> Optional[str]:
    """Identify one window; returns a normalised code, or None."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    stream = slid.create_stream()
    stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
    lang = slid.compute(stream)
    if not lang:
        return None
    # Some builds hand back the raw whisper token spelling, "<|de|>".
    code = str(lang).strip().strip("<|>").lower()
    return WHISPER_CODE_ALIASES.get(code, code) or None


def detect_language(
    windows: list[np.ndarray],
    encoder: Path,
    decoder: Path,
    *,
    num_threads: int = 1,
) -> Optional[str]:
    """Identify the spoken language from short PCM probe windows.

    Each window is 16kHz mono float32 decoded at speed 1.0 — time-stretched
    audio shifts the spectrum and degrades LID the same way it degrades
    diarization.

    Windows are identified independently. When they agree, that's the answer.
    When they don't, they are concatenated and identified once more as a
    tiebreak, which is steadier than trusting whichever window happened to come
    first. Returns None if nothing usable came back.
    """
    windows = [w for w in windows if w is not None and len(w)]
    if not windows:
        return None

    slid = _build_slid(encoder, decoder, num_threads)

    votes = [v for v in (_compute(slid, w) for w in windows) if v]
    if not votes:
        return None
    if len(set(votes)) == 1:
        return votes[0]

    return _compute(slid, np.concatenate(windows))
