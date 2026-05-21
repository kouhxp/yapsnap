#!/usr/bin/env python3
"""
yapsnap — snap any video or audio into plaintext.

Transcribes a local audio/video file or a URL (YouTube, X, TikTok, Instagram,
or any direct mp3/mp4 URL) using the sherpa-onnx streaming Kroko English
transducer. Runs entirely on CPU.

Usage:
    yapsnap INPUT [-o OUTPUT.txt] [--timestamps] [--speed 1.5] [--keep-audio] [--model DIR]

INPUT may be:
    - a local file (.mp3, .mp4, .m4a, .wav, .webm, ...)
    - a URL (YouTube, X/Twitter, TikTok, Instagram, generic mp4/mp3, ...)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
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

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


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


def _download(url: str, dest: Path) -> None:
    """Download a URL to a file, showing simple progress.

    Sends a User-Agent (HuggingFace sometimes rejects the default urllib UA).
    Sanity-checks tokens.txt as text and .onnx as binary > 1KB, so we fail loud
    instead of writing an HTML error page to disk.
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
    tmp.replace(dest)


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


def transcribe_sentences(samples: np.ndarray, model_dir: Path, speed: float
                         ) -> tuple[str, Optional[list[tuple[float, str]]]]:
    """Decode `samples` and return (plain_text, sentences).

    `sentences` is a list of (original_time_seconds, text) tuples when this
    sherpa-onnx build exposes alignment data, or None when it does not. The
    start times are already scaled back to original-audio time by `speed`.

    This is the structured core used by both plain transcription and the
    diarization path (which needs per-sentence start times for alignment).
    """
    import sherpa_onnx  # imported lazily so --help is fast

    if len(samples) == 0:
        return "", []

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=find_model_file(model_dir, "encoder"),
        decoder=find_model_file(model_dir, "decoder"),
        joiner=find_model_file(model_dir, "joiner"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=os.cpu_count() or 1,
        sample_rate=SAMPLE_RATE,
        feature_dim=FEATURE_DIM,
        decoding_method="greedy_search",
        provider="cpu",
        enable_endpoint_detection=False,
    )

    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    # Tail padding flushes the final chunk through the streaming model.
    tail_len = int(0.66 * SAMPLE_RATE)
    if tail_len > 0:
        stream.accept_waveform(SAMPLE_RATE, np.zeros(tail_len, dtype=np.float32))
    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    # In the Python wrapper, OnlineRecognizer.get_result(stream) returns a
    # plain str (just the text). Different versions expose token/timestamp
    # data through different paths; try each.
    text_result = recognizer.get_result(stream)
    text = (text_result if isinstance(text_result, str)
            else getattr(text_result, "text", "") or "").strip()

    toks, times = _try_extract_timestamps(recognizer, stream, text_result)
    if not toks or not times or len(toks) != len(times):
        return text, None  # no alignment available for this build

    sentences = _group_into_sentences(toks, [float(t) for t in times], speed)
    return text, sentences


def transcribe(samples: np.ndarray, model_dir: Path,
               timestamps: bool, speed: float) -> str:
    text, sentences = transcribe_sentences(samples, model_dir, speed)

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
        num_threads=os.cpu_count() or 1,
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
    ap.add_argument("--speed", type=float, default=1.5,
                    help="Audio speedup factor (default 1.5; pitch preserved).")
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
                                  timestamps=args.timestamps, speed=args.speed)
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
