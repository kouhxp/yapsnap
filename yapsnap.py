#!/usr/bin/env python3
"""
yapsnap — snap any video or audio into plaintext.

Transcribes a local audio/video file or a URL (YouTube, X, TikTok, Instagram,
or any direct mp3/mp4 URL) using the sherpa-onnx streaming Kroko English
transducer. Runs entirely on CPU.

Usage:
    yapsnap INPUT [-o OUTPUT.txt] [--timestamps] [--speed 1.4] [--keep-audio] [--model DIR]

INPUT may be:
    - a local file (.mp3, .mp4, .m4a, .wav, .webm, ...)
    - a URL (YouTube, X/Twitter, TikTok, Instagram, generic mp4/mp3, ...)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
FEATURE_DIM = 80

# Auto-download source for the default Kroko English streaming model.
# The repo ships plain .onnx (no .int8 variants); find_model_file still prefers
# int8 if a custom --model dir provides them.
DEFAULT_MODEL_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
DEFAULT_MODEL_FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")
HF_BASE = "https://huggingface.co"
HTTP_UA = "yapsnap/0.1 (+https://github.com/)"

# Security: SHA-256 manifest of known-good model digests. Ships in the package
# (see MANIFEST.in) and sits next to this module. _download() verifies every
# fetched file against it under a strict hard-fail policy:
#
#   - file in manifest, hash matches  -> accept
#   - file in manifest, hash MISMATCH -> hard fail (integrity error)
#   - file NOT in manifest            -> hard fail (unrecognised file)
#   - manifest missing / empty        -> hard fail (nothing can be verified)
#
# There is no opt-out. This gates auto-downloads only; a model passed via
# --model / KROKO_MODEL loads from disk without going through _download(), so
# custom local models are unaffected. To auto-download a new model, add its
# digest to the manifest first (scripts/gen_hashes.sh --models).
MODEL_CHECKSUMS_FILE = Path(__file__).resolve().parent / "model_checksums.sha256"

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CPU thread budget
# ---------------------------------------------------------------------------

def _thread_budget() -> int:
    """Total inference-thread budget = physical cores (clamped to affinity).

    Delegates to cpuopt.CpuPlan, which detects physical cores across Linux /
    macOS / Windows and honours cgroup/taskset limits. cpuopt is part of
    yapsnap, but if it is somehow unimportable we fall back to a conservative
    SMT-aware guess so the tool still runs.
    """
    try:
        import cpuopt
        return max(1, cpuopt.get_plan().num_threads)
    except Exception:
        logical = os.cpu_count() or 1
        # Assume 2-way SMT on even counts >= 4; otherwise trust logical.
        if logical >= 4 and logical % 2 == 0:
            return logical // 2
        return logical


# ---------------------------------------------------------------------------
# Model resolution / auto-download
# ---------------------------------------------------------------------------

def user_cache_dir() -> Path:
    """Return a per-user cache dir for storing the auto-downloaded model.
    Cross-platform: XDG on Linux, ~/Library/Caches on macOS, %LOCALAPPDATA% on Windows.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "yapsnap"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "yapsnap"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "yapsnap"


def find_model_file(model_dir: Path, base: str) -> str:
    """Find encoder/decoder/joiner ONNX, preferring int8."""
    for name in (f"{base}.int8.onnx", f"{base}.onnx"):
        p = model_dir / name
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"No {base}(.int8).onnx found in {model_dir}")


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------
#
# Policy: every auto-downloaded model file MUST have a matching SHA-256 digest
# in model_checksums.sha256. There is no opt-out and no lenient mode:
#
#   - file in manifest, hash matches  -> accept
#   - file in manifest, hash differs  -> hard fail (integrity error)
#   - file NOT in manifest            -> hard fail (unrecognised file)
#   - manifest missing entirely       -> hard fail (cannot verify anything)
#
# This gates ONLY auto-downloads (the _download() path). A model supplied with
# --model / KROKO_MODEL is loaded straight from disk and never passes through
# here, so custom local models are unaffected. The practical consequence: to
# auto-download a NEW model (e.g. another Kroko language), its digest must be
# added to model_checksums.sha256 first. Until then, fetch it manually and pass
# the directory with --model.


def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file, read in chunks so
    large model files don't have to sit in memory all at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model_checksums() -> dict[str, str]:
    """Parse model_checksums.sha256 into {filename: sha256-hex}.

    Format is the standard `sha256sum` layout, one entry per line:

        <64-hex-digest>  <filename>

    Blank lines and `#` comments are ignored. Only the basename of the second
    field is kept, so a digest written for `diarization-models/foo.onnx` still
    matches a download whose dest is `<cache>/foo.onnx`. A missing or
    unreadable manifest yields an empty dict; verify_checksum() turns that into
    a hard failure rather than silently trusting the download.
    """
    checksums: dict[str, str] = {}
    if not MODEL_CHECKSUMS_FILE.is_file():
        return checksums
    try:
        text = MODEL_CHECKSUMS_FILE.read_text(encoding="utf-8")
    except OSError:
        return checksums
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].strip().lower(), parts[1].strip()
        # Tolerate the `*filename` binary-mode marker that sha256sum may emit.
        if name.startswith("*"):
            name = name[1:]
        if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
            checksums[os.path.basename(name)] = digest
    return checksums


def verify_checksum(path: Path) -> None:
    """Verify a downloaded file against the SHA-256 manifest, or raise.

    Hard-fail policy (see the module comment above): a missing manifest, an
    unlisted file, or a digest mismatch all raise RuntimeError. There is no
    success path that skips verification.

    Raising happens AFTER the caller has written the file to a temp path but
    BEFORE it is promoted to its final name, so a rejected file never lands in
    the cache.
    """
    checksums = load_model_checksums()

    # A missing manifest means nothing can be verified. Refuse outright rather
    # than trusting the download.
    if not checksums:
        if not MODEL_CHECKSUMS_FILE.is_file():
            raise RuntimeError(
                f"checksum manifest {MODEL_CHECKSUMS_FILE} is missing \u2014 "
                f"cannot verify model integrity, refusing to proceed. "
                f"Reinstall yapsnap or restore the manifest."
            )
        raise RuntimeError(
            f"checksum manifest {MODEL_CHECKSUMS_FILE} contains no usable "
            f"digests \u2014 cannot verify model integrity, refusing to proceed."
        )

    name = path.name
    expected = checksums.get(name)

    if expected is None:
        raise RuntimeError(
            f"{name} is not listed in {MODEL_CHECKSUMS_FILE.name}; yapsnap "
            f"refuses to use an unrecognised auto-downloaded model. If this is "
            f"a model you trust, add its sha256 digest to the manifest "
            f"(scripts/gen_hashes.sh --models), or fetch it yourself and pass "
            f"its directory with --model."
        )

    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"checksum mismatch for {name}:\n"
            f"  expected sha256 {expected}\n"
            f"  got      sha256 {actual}\n"
            f"The download may be corrupt or tampered with; refusing to use it."
        )
    print(f"  verified {name} (sha256 ok)", file=sys.stderr)


def _download(url: str, dest: Path) -> None:
    """Download a URL to a file, showing simple progress.

    Sends a User-Agent (HuggingFace sometimes rejects the default urllib UA).
    Sanity-checks tokens.txt as text and .onnx as binary > 1KB, so we fail loud
    instead of writing an HTML error page to disk. Finally verifies the file
    against the SHA-256 manifest before promoting it into place.
    """
    print(f"  fetching {url}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.reason} for {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error for {url}: {e.reason}") from e

    with resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 16
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = 100.0 * read / total
                print(f"\r    {read/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.0f}%)",
                      end="", file=sys.stderr)
        print("", file=sys.stderr)

    # Sanity check: don't accept tiny HTML error pages as ONNX files.
    size = tmp.stat().st_size
    if dest.suffix == ".onnx" and size < 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded {dest.name} is only {size} bytes \u2014 likely an error page, not a model"
        )

    # Integrity check against the SHA-256 manifest. verify_checksum() inspects
    # by final name (dest.name), so the temp `.part` file is checked as what it
    # will become. On failure, drop the temp file so a bad download never
    # lingers in the cache or gets promoted.
    verify_target = tmp.with_name(dest.name)
    tmp.rename(verify_target)
    try:
        verify_checksum(verify_target)
    except Exception:
        verify_target.unlink(missing_ok=True)
        raise
    verify_target.replace(dest)


def ensure_default_model() -> Path:
    """Download the default Kroko English model into the user cache if missing.
    Returns the directory containing encoder/decoder/joiner/tokens."""
    cache = user_cache_dir() / DEFAULT_MODEL_REPO.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    missing = [f for f in DEFAULT_MODEL_FILES if not (cache / f).is_file()]
    if not missing:
        return cache

    print(f"Downloading Kroko English model to {cache}", file=sys.stderr)
    for fname in missing:
        url = f"{HF_BASE}/{DEFAULT_MODEL_REPO}/resolve/main/{fname}"
        try:
            _download(url, cache / fname)
        except Exception as e:
            raise RuntimeError(
                f"failed to download {fname} from {url}: {e}\n"
                f"You can download the model manually from {HF_BASE}/{DEFAULT_MODEL_REPO} "
                f"and pass its directory via --model."
            ) from e
    return cache


def resolve_model_dir(model_arg: Optional[Path]) -> Path:
    """Resolve the model directory from CLI arg, env var, or auto-download."""
    candidate = model_arg or (
        Path(os.environ["KROKO_MODEL"]) if os.environ.get("KROKO_MODEL") else None
    )
    if candidate is not None:
        if not candidate.is_dir():
            raise FileNotFoundError(f"--model directory not found: {candidate}")
        if not (candidate / "tokens.txt").is_file():
            raise FileNotFoundError(f"tokens.txt not found in {candidate}")
        return candidate
    return ensure_default_model()


# ---------------------------------------------------------------------------
# Input handling: download URLs, decode any media to PCM
# ---------------------------------------------------------------------------

def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"required tool '{name}' not found in PATH. "
            f"Install it and try again (see README)."
        )


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def download_url(url: str, workdir: Path) -> Path:
    """Use yt-dlp to download audio from any supported URL.

    Format selection is tuned to avoid TikTok's audio-stripped 'Playback video'
    formats. The chain tries, in order:

      1. download_addr   \u2014 TikTok's regular download format (always carries audio)
      2. h264            \u2014 TikTok's 'Direct video' h264 format (also has audio)
      3. bestaudio       \u2014 audio-only stream when offered (YouTube etc.)
      4. best            \u2014 generic fallback

    The first two are TikTok-specific format IDs; on every other site they
    simply don't match and yt-dlp falls through to bestaudio/best cleanly.

    Returns path to a single media file in workdir.
    """
    _require_tool("yt-dlp")
    workdir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(workdir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "-f", "download_addr/h264/bestaudio/best",
        "-o", out_tpl,
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed for {url}: exit {e.returncode}") from e

    files = sorted(p for p in workdir.iterdir() if p.is_file())
    if not files:
        raise RuntimeError(f"yt-dlp produced no files for {url}")
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[0]


def _has_audio_stream(media_path: Path) -> bool:
    """Return True if ffprobe finds at least one audio stream."""
    if shutil.which("ffprobe") is None:
        # ffprobe ships with ffmpeg; if it's missing, skip the check.
        return True
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(media_path)],
            capture_output=True, check=True, text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return "audio" in proc.stdout.lower()


def decode_to_pcm(media_path: Path, speed: float = 1.0) -> np.ndarray:
    """Decode any audio/video file ffmpeg understands to 16kHz mono float32.
    `speed` applies atempo (pitch-preserving). atempo accepts 0.5..2.0 per stage.
    """
    _require_tool("ffmpeg")

    if not _has_audio_stream(media_path):
        raise RuntimeError(
            f"{media_path.name} has no audio stream that ffmpeg can read. "
            f"For TikTok this can happen when only the 'Playback video' format was "
            f"available; updating yt-dlp (pip install -U yt-dlp) usually fixes it."
        )

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(media_path)]
    if abs(speed - 1.0) > 1e-5:
        stages, remaining = [], float(speed)
        while remaining > 2.0:
            stages.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            stages.append("atempo=0.5")
            remaining /= 0.5
        if abs(remaining - 1.0) > 1e-5:
            stages.append(f"atempo={remaining}")
        if stages:
            cmd += ["-filter:a", ",".join(stages)]

    cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed for {media_path}: {e.stderr.decode('utf-8', errors='ignore')}"
        ) from e

    if not proc.stdout:
        return np.array([], dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _format_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _join_tokens(tokens: list[str]) -> str:
    """Reconstruct text from BPE-style tokens that use ▁ for word starts."""
    out = []
    for t in tokens:
        if t.startswith("\u2581"):
            out.append(" " + t[1:])
        else:
            out.append(t)
    return "".join(out).strip()


def _group_into_sentences(tokens: list[str], timestamps: list[float], speed: float
                          ) -> list[tuple[float, str]]:
    """Group tokens into sentences ending in .!? and tag each sentence with its
    start time, scaled by `speed` to map back to original-audio time.
    """
    sentences: list[tuple[float, str]] = []
    buf_tokens: list[str] = []
    buf_start: Optional[float] = None
    end_chars = (".", "!", "?")

    for tok, ts in zip(tokens, timestamps):
        if not buf_tokens:
            buf_start = ts
        buf_tokens.append(tok)
        if tok and tok[-1] in end_chars:
            text = _join_tokens(buf_tokens)
            if text:
                sentences.append(((buf_start or 0.0) * speed, text))
            buf_tokens = []
            buf_start = None

    # Flush any trailing tokens with no terminal punctuation.
    if buf_tokens:
        text = _join_tokens(buf_tokens)
        if text:
            sentences.append(((buf_start or 0.0) * speed, text))
    return sentences


def _build_recognizer(model_dir: Path, num_threads: int):
    """Construct the sherpa-onnx OnlineRecognizer for the Kroko transducer."""
    import sherpa_onnx
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=find_model_file(model_dir, "encoder"),
        decoder=find_model_file(model_dir, "decoder"),
        joiner=find_model_file(model_dir, "joiner"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method="greedy_search",
        provider="cpu",
        enable_endpoint_detection=False,
    )


def transcribe_sentences(
    samples: np.ndarray,
    model_dir: Path,
    speed: float,
    *,
    num_workers: int = 0,
    threads_per_worker: int = 0,
) -> tuple[str, Optional[list[tuple[float, str]]]]:
    """Decode `samples` and return (plain_text, sentences).

    `sentences` is a list of (original_time_seconds, text) tuples when this
    sherpa-onnx build exposes alignment data, or None when it does not. The
    start times are already scaled back to original-audio time by `speed`.

    Parallel decoding
    -----------------
    When `num_workers` > 1 the audio is split into `num_workers` overlapping
    chunks, each decoded on its own thread with the ordinary single-stream
    path. ONNX Runtime releases the GIL during inference, so the chunks decode
    concurrently. This is the `--workers N --threads M` mechanism.

    `threads_per_worker` sets the ONNX intra-op thread count for the shared
    recognizer (GEMM parallelism within each frame).

    Defaults (0 = autodetect):
        num_workers  : 1 on ≤ 2 cores, else min(4, physical_cores)
        threads_per_worker : max(1, logical_cores // num_workers)

    Overlap
    -------
    Adjacent chunks overlap by 1 second of audio (SAMPLE_RATE samples).
    After decoding, tokens whose timestamps fall inside the overlap zone are
    discarded from the *later* chunk, so the earlier chunk "owns" the boundary.
    """
    if len(samples) == 0:
        return "", []

    # --- auto-detect workers / threads ---
    #
    # The governing rule, from benchmarking on an 8-logical / 4-physical box:
    #
    #   * SPEED is set by TOTAL threads (workers x threads_per_worker), and the
    #     sweet spot is total threads ~= PHYSICAL cores. Going past physical
    #     cores (onto hyperthread siblings) made every run SLOWER, not faster:
    #       workers 2 x threads 2  (4 total) -> ~4m06   <- best
    #       workers 2 x threads 4  (8 total) -> ~7m43
    #       workers 1 x threads 8  (8 total) -> ~12m
    #       workers 4 x threads 8  (32 total, old default) -> ~12m (thrashing)
    #
    #   * Splitting that budget across a few workers (~2 threads each) was as
    #     fast as or faster than one fat single stream, and read better.
    #
    # So: take the physical-core thread budget from cpuopt, then split it into
    # workers of ~THREADS_PER_WORKER_TARGET threads each.
    THREADS_PER_WORKER_TARGET = 2

    budget = _thread_budget()  # physical cores (cpuopt), clamped to affinity

    if threads_per_worker <= 0:
        threads_per_worker = min(THREADS_PER_WORKER_TARGET, budget)

    if num_workers <= 0:
        if budget <= 2:
            # 1-2 physical cores: one stream, use the whole budget. Splitting
            # here only adds warmup/process overhead with no cores to fill.
            num_workers = 1
            threads_per_worker = budget
        else:
            num_workers = max(1, budget // threads_per_worker)

    # Each non-first chunk pays a 15s warmup overlap that is decoded but
    # discarded. For the split to be worthwhile a chunk must own meaningfully
    # more audio than it wastes on warmup — require >= 45s of owned content,
    # i.e. 3x the warmup. Below that, drop a worker.
    min_owned_samples = SAMPLE_RATE * 45
    if len(samples) < min_owned_samples * num_workers:
        num_workers = max(1, len(samples) // min_owned_samples)

    if num_workers <= 1:
        # Single stream gets the whole thread budget, not just one worker's
        # share — there are no sibling workers to share cores with.
        return _transcribe_single(samples, model_dir, speed, budget)

    return _transcribe_batched(
        samples, model_dir, speed, num_workers, threads_per_worker,
    )


def _transcribe_single(
    samples: np.ndarray, model_dir: Path, speed: float, num_threads: int,
) -> tuple[str, Optional[list[tuple[float, str]]]]:
    """Original single-stream decode path."""
    recognizer = _build_recognizer(model_dir, num_threads)

    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    tail_len = int(0.66 * SAMPLE_RATE)
    if tail_len > 0:
        stream.accept_waveform(SAMPLE_RATE, np.zeros(tail_len, dtype=np.float32))
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text_result = recognizer.get_result(stream)
    text = (text_result if isinstance(text_result, str)
            else getattr(text_result, "text", "") or "").strip()

    toks, times = _try_extract_timestamps(recognizer, stream, text_result)
    if not toks or not times or len(toks) != len(times):
        return text, None

    sentences = _group_into_sentences(toks, [float(t) for t in times], speed)
    return text, sentences


def _decode_one_chunk(model_dir: Path, num_threads: int,
                      chunk: np.ndarray,
                      ) -> tuple[list[str], list[float]]:
    """Decode one pre-sliced audio chunk via the single-stream path.

    This runs in a SEPARATE PROCESS (see _transcribe_batched). Process
    isolation is not an optimization here — it is required for correctness.

    sherpa-onnx's feature extractor applies optional dithering, a randomized
    noise floor whose RNG state lives at process scope. Several feature
    extractors running concurrently in one process draw from that shared RNG
    in a timing-dependent order, so a chunk's features — and therefore its
    transcription — depend on the race with sibling chunks. ONNX Runtime also
    keeps process-global env/threadpool state. A child process has its own
    RNG and its own ONNX env, so this chunk decodes byte-for-byte identically
    to a standalone single-stream run of the same audio.

    Returns (tokens, timestamps) with timestamps in encoder-frame seconds
    relative to the start of this chunk (NOT yet offset or speed-scaled).
    """
    tail_len = int(0.66 * SAMPLE_RATE)

    recognizer = _build_recognizer(model_dir, num_threads)

    chunk = np.ascontiguousarray(chunk, dtype=np.float32)

    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, chunk)
    if tail_len > 0:
        stream.accept_waveform(SAMPLE_RATE,
                               np.zeros(tail_len, dtype=np.float32))
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text_result = recognizer.get_result(stream)
    toks, times = _try_extract_timestamps(recognizer, stream, text_result)
    if not toks or not times or len(toks) != len(times):
        return [], []
    return list(toks), [float(t) for t in times]


def _transcribe_batched(
    samples: np.ndarray,
    model_dir: Path,
    speed: float,
    num_workers: int,
    threads_per_worker: int,
) -> tuple[str, Optional[list[tuple[float, str]]]]:
    """Split audio into overlapping chunks and decode them in parallel
    PROCESSES.

    Each chunk is decoded in its own child process via the ordinary
    single-stream ``decode_stream`` path (see _decode_one_chunk). The pool
    uses the 'spawn' start method: a 'fork'ed child inherits the parent's
    native-library state (ONNX Runtime threadpool, OpenMP), which does not
    survive fork and corrupts inference. 'spawn' gives each child a clean
    interpreter that initialises ONNX Runtime itself.

    ``decode_streams`` (batched GEMM) was tried and rejected: it is not
    bit-identical to single-stream decoding for this INT8 Zipformer model.

    Encoder-warmup overlap
    ----------------------
    The Zipformer encoder is stateful: each layer caches `left_context_len`
    frames of attention key/value, zero-filled on a fresh stream. Each
    non-first chunk therefore starts OVERLAP_SECONDS early so that, by the
    time it reaches the audio it "owns", the encoder cache holds real frames.
    Tokens whose timestamps land in the warmup zone are dropped from the later
    chunk — the earlier chunk owns that audio.
    """
    total_samples = len(samples)
    OVERLAP_SECONDS = 15
    overlap = SAMPLE_RATE * OVERLAP_SECONDS

    # Chunk boundaries. `end` is the owned boundary; `start` is pulled back by
    # `overlap` for warmup on every chunk after the first.
    chunk_len = (total_samples + num_workers - 1) // num_workers
    chunks: list[tuple[int, int]] = []
    for i in range(num_workers):
        owned_start = i * chunk_len
        start = owned_start if i == 0 else max(0, owned_start - overlap)
        end = min(total_samples, (i + 1) * chunk_len)
        if owned_start >= total_samples:
            break
        chunks.append((start, end))

    # Decode every chunk in its own process. Each chunk's PCM is sliced out
    # (an owned, contiguous copy) and pickled to the child. results[i] =
    # (tokens, chunk_local_times).
    #
    # The pool MUST use the 'spawn' start method, not the Linux default
    # 'fork'. A forked child inherits the parent's address space, including
    # the state of native libraries (ONNX Runtime, its threadpool, OpenMP).
    # Those threadpools do not survive fork: a forked child calling back into
    # ONNX Runtime gets subtly corrupted intra-op state, which perturbs the
    # INT8 GEMM results just enough to change transcription. 'spawn' starts a
    # clean interpreter that imports and initialises everything fresh.
    import multiprocessing as _mp
    mp_ctx = _mp.get_context("spawn")

    results: list[tuple[list[str], list[float]]] = [([], [])] * len(chunks)
    with ProcessPoolExecutor(max_workers=len(chunks),
                             mp_context=mp_ctx) as pool:
        futures = {}
        for idx, (start, end) in enumerate(chunks):
            chunk = np.ascontiguousarray(
                samples[start:end], dtype=np.float32).copy()
            fut = pool.submit(_decode_one_chunk, model_dir,
                              threads_per_worker, chunk)
            futures[fut] = idx
        for fut in futures:
            idx = futures[fut]
            results[idx] = fut.result()

    # Stitch: convert each chunk's local timestamps to original-audio seconds
    # (add the chunk's start offset, then scale by speed), and drop tokens that
    # fall inside the warmup overlap of non-first chunks.
    all_toks: list[str] = []
    all_times: list[float] = []
    has_timestamps = True

    for idx, (start, end) in enumerate(chunks):
        toks, local_times = results[idx]
        if not toks or not local_times:
            has_timestamps = False
            continue

        offset_seconds = start / SAMPLE_RATE
        times = [(t + offset_seconds) * speed for t in local_times]

        if idx == 0:
            all_toks.extend(toks)
            all_times.extend(times)
        else:
            cutoff = all_times[-1] if all_times else 0.0
            for tok, t in zip(toks, times):
                if t > cutoff:
                    all_toks.append(tok)
                    all_times.append(t)

    if not has_timestamps or not all_toks:
        raise RuntimeError(
            "batched decode produced no timestamp data; rerun with --workers 1"
        )

    text = _join_tokens(all_toks)
    sentences = _group_into_sentences(all_toks, all_times, speed)
    return text, sentences


def transcribe(samples: np.ndarray, model_dir: Path,
               timestamps: bool, speed: float,
               num_workers: int = 0, threads_per_worker: int = 0) -> str:
    text, sentences = transcribe_sentences(
        samples, model_dir, speed,
        num_workers=num_workers, threads_per_worker=threads_per_worker,
    )

    if not timestamps:
        return text

    if sentences is None:
        # No alignment data available for this build of sherpa-onnx; fall
        # back to plain text and warn the user once.
        print("warning: this sherpa-onnx build did not return timestamp data; "
              "emitting plain text instead", file=sys.stderr)
        return text

    return "\n".join(f"[{_format_mmss(t)}] {s}" for t, s in sentences)


def transcribe_diarized(media_path: Path, model_dir: Path, speed: float,
                        diarize_model: str, num_speakers: int) -> str:
    """Transcribe with speaker labels.

    ASR runs on the sped-up signal (fast); diarization runs on the SAME audio
    decoded at 1.0x (sped-up audio wrecks segmentation + embeddings). Both
    timelines live in original-audio seconds, so sentence start times and
    speaker turns share one clock.

    Output lines look like:  SPEAKER_00 [MM:SS]: text
    """
    import diarize as diar

    # 1) ASR on sped-up audio -> sentences with original-time start seconds.
    sped = decode_to_pcm(media_path, speed=speed)
    _text, sentences = transcribe_sentences(sped, model_dir, speed)
    if sentences is None:
        raise RuntimeError(
            "diarization needs sentence timestamps, but this sherpa-onnx build "
            "did not return alignment data. Try a build that supports "
            "timestamps, or run without --diarize."
        )
    if not sentences:
        return ""

    # 2) Diarization on original-speed audio.
    cache = user_cache_dir() / "diarization-models"
    seg_onnx, emb_onnx = diar.ensure_diarization_models(
        diarize_model, cache, _download
    )
    pcm_1x = sped if abs(speed - 1.0) < 1e-5 else decode_to_pcm(media_path, speed=1.0)
    turns = diar.diarize_pcm(
        pcm_1x, seg_onnx, emb_onnx,
        num_speakers=num_speakers,
        num_threads=_thread_budget(),
        show_progress=True,
    )

    # 3) Attach a speaker to each sentence and format.
    labeled = diar.label_sentences(sentences, turns)
    lines = []
    for spk, t, text in labeled:
        label = f"SPEAKER_{spk:02d}" if spk is not None else "SPEAKER_??"
        lines.append(f"{label} [{_format_mmss(t)}]: {text}")
    return "\n".join(lines)


def _try_extract_timestamps(recognizer, stream, text_result):
    """Best-effort extraction of (tokens, timestamps) across sherpa-onnx
    Python wrapper versions. Returns ([], []) if nothing usable is found.
    """
    import json

    # Path 1: recognizer.get_result_as_json_string(stream) (C-API parity)
    json_method = getattr(recognizer, "get_result_as_json_string", None)
    if callable(json_method):
        try:
            parsed = json.loads(json_method(stream))
            toks = list(parsed.get("tokens") or [])
            times = list(parsed.get("timestamps") or [])
            if toks and times:
                return toks, times
        except Exception:
            pass

    # Path 2: stream.result.tokens / stream.result.timestamps
    sresult = getattr(stream, "result", None)
    if sresult is not None:
        toks = list(getattr(sresult, "tokens", None) or [])
        times = list(getattr(sresult, "timestamps", None) or [])
        if toks and times:
            return toks, times

    # Path 3: text_result is itself a struct (some wrappers wrap a struct)
    if not isinstance(text_result, str) and text_result is not None:
        toks = list(getattr(text_result, "tokens", None) or [])
        times = list(getattr(text_result, "timestamps", None) or [])
        if toks and times:
            return toks, times

    return [], []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_name(input_arg: str) -> str:
    """Pick a default output filename in cwd."""
    if is_url(input_arg):
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(input_arg)
        stem: Optional[str] = None

        # Prefer ?v=<id> on YouTube watch URLs.
        if "v" in parse_qs(parsed.query):
            stem = parse_qs(parsed.query)["v"][0]
        else:
            # For sites like x.com/user/status/<id>/video/1, the last path segment
            # is often "1"; pick the longest path segment instead (usually the ID).
            segments = [s for s in parsed.path.split("/") if s]
            if segments:
                stem = max(segments, key=len)
                # If it has a media extension (e.g. clip.mp4), drop it.
                stem = re.sub(r"\.(mp3|mp4|m4a|wav|webm|mov|mkv|aac|opus|ogg|flac)$",
                              "", stem, flags=re.IGNORECASE)

        stem = (stem or "transcript").strip()
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem) or "transcript"
        return f"{stem}_transcript.txt"
    return f"{Path(input_arg).stem}_transcript.txt"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="yapsnap",
        description="Snap a local audio/video file or any video URL into a "
                    "plaintext transcript using sherpa-onnx (Kroko English streaming).",
    )
    ap.add_argument("input", help="Local file path or URL (YouTube, X, TikTok, Instagram, mp4/mp3 URL).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .txt path (default: ./transcripts/<input>_transcript.txt).")
    ap.add_argument("--timestamps", action="store_true",
                    help="Include [MM:SS] sentence-level timestamps in output.")
    ap.add_argument("--speed", type=float, default=1.4,
                    help="Audio speedup factor (default 1.4; pitch preserved). "
                         "Lower reads cleaner at little/no time cost; raise for "
                         "a small speed gain on slower machines.")
    ap.add_argument("--keep-audio", action="store_true",
                    help="Keep the downloaded/intermediate audio file instead of deleting it.")
    ap.add_argument("--model", type=Path, default=None,
                    help="Path to a sherpa-onnx streaming transducer model directory "
                         "(or set KROKO_MODEL). Default: auto-download Kroko English.")
    ap.add_argument("--diarize", action="store_true",
                    help="Label speakers ('who spoke when'). Output lines become "
                         "'SPEAKER_00 [MM:SS]: text'. Implies --timestamps. CPU-only, "
                         "auto-downloads diarization models on first use.")
    ap.add_argument("--diarize-model", choices=["pyannote", "reverb"], default="pyannote",
                    help="Segmentation model for --diarize. 'pyannote' (CC-BY-4.0, "
                         "default) or 'reverb' (more accurate, NON-COMMERCIAL license).")
    ap.add_argument("--num-speakers", type=int, default=-1,
                    help="Known speaker count for --diarize (default -1 = auto-detect). "
                         "Set this when you know the count; auto-detection degrades "
                         "above ~7 speakers.")
    ap.add_argument("--workers", type=int, default=0, metavar="N",
                    help="Number of chunks to decode in parallel processes. "
                         "Default 0 = autodetect: split the physical-core "
                         "thread budget into workers of ~2 threads each. "
                         "Set 1 to disable chunking (single stream).")
    ap.add_argument("--threads", type=int, default=0, metavar="N",
                    help="ONNX intra-op threads per worker. Default 0 = "
                         "autodetect (~2). Total threads across all workers is "
                         "kept near the physical-core count; exceeding it "
                         "(e.g. onto hyperthread siblings) slows decoding down.")
    args = ap.parse_args(argv)

    if args.speed <= 0:
        print("error: --speed must be > 0", file=sys.stderr)
        return 2

    # Resolve the model first so we fail fast if it can't be obtained.
    try:
        model_dir = resolve_model_dir(args.model)
    except Exception as e:
        print(f"model error: {e}", file=sys.stderr)
        return 1

    # Resolve input -> a local media path.
    tmp_root: Optional[Path] = None
    cleanup_root: Optional[Path] = None
    try:
        if is_url(args.input):
            tmp_root = Path(tempfile.mkdtemp(prefix="transcribe-"))
            cleanup_root = None if args.keep_audio else tmp_root
            try:
                media_path = download_url(args.input, tmp_root)
            except Exception as e:
                print(f"download error: {e}", file=sys.stderr)
                return 1
            if args.keep_audio:
                print(f"audio kept at: {media_path}", file=sys.stderr)
        else:
            media_path = Path(args.input)
            if not media_path.is_file():
                print(f"error: file not found: {media_path}", file=sys.stderr)
                return 1

        try:
            if args.diarize:
                text = transcribe_diarized(
                    media_path, model_dir, speed=args.speed,
                    diarize_model=args.diarize_model,
                    num_speakers=args.num_speakers,
                )
            else:
                samples = decode_to_pcm(media_path, speed=args.speed)
                text = transcribe(samples, model_dir,
                                  timestamps=args.timestamps, speed=args.speed,
                                  num_workers=args.workers,
                                  threads_per_worker=args.threads)
        except Exception as e:
            print(f"transcription error: {e}", file=sys.stderr)
            return 1

        out_path = args.output if args.output else (
            Path.cwd() / "transcripts" / _default_output_name(args.input)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(text + "\n", encoding="utf-8")
        except Exception as e:
            print(f"error writing output: {e}", file=sys.stderr)
            return 1

        print(out_path)
        return 0
    finally:
        if cleanup_root is not None and cleanup_root.is_dir():
            shutil.rmtree(cleanup_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
