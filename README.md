# yapsnap

> **Snap any video URL or audio file into plaintext. No GPU. No cloud. One command.**

![Python](https://img.shields.io/badge/python-3.9+-blue) ![License](https://img.shields.io/badge/license-Apache--2.0-green) ![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

```bash
yapsnap "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

That's it. You get a `.txt` next to your shell, transcribed on your CPU, in less time than it took the video to play.

---

## Why yapsnap

- ⚡ **Fast on CPU.** Streaming Zipformer transducer (Kroko English) chews through audio at several times realtime on a laptop. No CUDA. No M-series-only tricks. Plain old cores.
- 🌐 **Any video URL, plus local files.** YouTube. X. TikTok. Instagram Reels. Direct `.mp4`/`.mp3` links. Or just point it at a file on disk. yt-dlp handles the fetch, ffmpeg handles the decode, the rest is yours.
- 📴 **Offline after first run.** ~80 MB model downloads once to your cache and stays there. No API keys. No quotas. Your audio never leaves your machine.
- 🪶 **Lean deps.** `sherpa-onnx`, `numpy`, `yt-dlp` — that's the whole runtime, diarization included. No PyTorch, no cloud SDKs.
- 🗣 **Ten-plus languages.** English out of the box; French, German, Spanish, Italian, Portuguese, Dutch, Swedish, Swiss German, Hebrew, and Turkish are a one-line `--model` swap away. See [Other languages](#other-languages).
- ⏱ **Sentence-level timestamps when you want them.** `--timestamps` adds `[MM:SS]` per sentence using Kroko's built-in punctuation. Timing stays correct even when you transcribe at 2x.
- 🗣️ **Speaker labels, optional.** `--diarize` answers "who spoke when" and prefixes each line with `SPEAKER_00`, `SPEAKER_01`, … Still CPU-only, still ONNX — no PyTorch, no extra runtime deps. See [Diarization](#diarization).

---

## Quickstart

```bash
# 1. ffmpeg on PATH (one-time, per OS — see below)
# 2. Install (from PyPI, or `pip install .` from a clone)
pip install yapsnap

# 3. Snap something
yapsnap https://www.tiktok.com/@user/video/7234567890123456789
yapsnap meeting.mp4 --timestamps
yapsnap interview.mp3 --diarize          # label speakers
yapsnap podcast.mp3 -o ~/notes/episode.txt
```

The first run downloads the model (~80 MB). Every run after is offline.

---

## What it handles

Any URL `yt-dlp` understands works. The big ones:

| Source            | Example                                                 |
|-------------------|---------------------------------------------------------|
| YouTube           | `https://www.youtube.com/watch?v=...`                   |
| YouTube Shorts    | `https://www.youtube.com/shorts/...`                    |
| X / Twitter       | `https://x.com/user/status/.../video/1`                 |
| TikTok            | `https://www.tiktok.com/@user/video/...`                |
| Instagram Reels   | `https://www.instagram.com/reel/.../`                   |
| Direct media URL  | `https://example.com/clip.mp4`                          |

Plus any local file ffmpeg can decode: `.mp3`, `.mp4`, `.m4a`, `.wav`, `.webm`, `.mov`, `.mkv`, `.aac`, `.opus`, `.ogg`, `.flac`, and friends.

---

## Install

### 1. ffmpeg

| OS      | Command                                                  |
|---------|----------------------------------------------------------|
| macOS   | `brew install ffmpeg`                                    |
| Linux   | `sudo apt install ffmpeg` *or* `sudo dnf install ffmpeg` |
| Windows | `winget install ffmpeg` *or* `choco install ffmpeg`      |

### 2. yapsnap

From PyPI (recommended):

```bash
pip install yapsnap
```

From source:

```bash
git clone https://github.com/kouhxp/yapsnap
cd yapsnap
pip install .
```

Installs two equivalent commands on your `PATH`: **`yapsnap`** (canonical) and **`transcribe`** (alias, for when the name slips your mind).

---

## Usage

```bash
# Local file
yapsnap path/to/audio.mp3

# Any video URL
yapsnap "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Sentence-level timestamps
yapsnap input.mp4 --timestamps

# Speaker labels ("who spoke when")
yapsnap interview.mp3 --diarize

# Speaker labels with a known speaker count (more reliable than auto-detect)
yapsnap call.mp3 --diarize --num-speakers 2

# Custom output path
yapsnap input.mp4 -o ./transcripts/talk.txt

# Don't speed audio up before transcribing (default is 1.4x, pitch preserved)
yapsnap input.mp4 --speed 1.0

# Keep the downloaded audio (URL inputs only)
yapsnap "https://..." --keep-audio
```

---

## Output

Plaintext, UTF-8. Default location is `./transcripts/` (created if missing) under the current working directory; override with `-o`. For URL inputs the filename is derived from the video ID (`dQw4w9WgXcQ_transcript.txt`, etc.).

**Without `--timestamps`** — one paragraph of recognized text:

```
Welcome to the show. Today we're talking about transcription. Let's get started.
```

**With `--timestamps`** — one sentence per line, timed against the original audio:

```
[00:00] Welcome to the show.
[00:03] Today we're talking about transcription.
[00:08] Let's get started.
```

Timestamps stay in original-audio time even at `--speed 1.4` or higher.

**With `--diarize`** — one sentence per line, each tagged with a speaker and timestamp:

```
SPEAKER_00 [00:00]: Welcome to the show.
SPEAKER_01 [00:03]: Glad to be here, thanks for having me.
SPEAKER_00 [00:08]: Let's get started.
```

Speaker numbers are assigned in order of appearance and are stable within a single run, but they carry no identity across files — `SPEAKER_00` in one transcript is unrelated to `SPEAKER_00` in another.

---

## Flags

| Flag              | Description                                                          |
|-------------------|----------------------------------------------------------------------|
| `-o`, `--output`  | Output `.txt` path. Default: `./transcripts/<input>_transcript.txt`. |
| `--timestamps`    | Emit `[MM:SS] sentence.` lines instead of a single paragraph.        |
| `--diarize`       | Label speakers (`SPEAKER_00 [MM:SS]: …`). Implies `--timestamps`.     |
| `--diarize-model` | Segmentation model: `pyannote` (default) or `reverb`. See below.     |
| `--num-speakers`  | Known speaker count for `--diarize`. Default `-1` (auto-detect).     |
| `--speed`         | Pre-transcription speedup factor, pitch preserved. Default `1.4`.    |
| `--workers`       | Chunks to decode in parallel processes. Default `0` (autodetect).    |
| `--threads`       | ONNX threads per worker. Default `0` (autodetect). See [Performance](#performance). |
| `--keep-audio`    | Keep the downloaded audio (URL inputs only).                         |
| `--model`         | Override the model directory. Also reads `KROKO_MODEL` env var.      |

---

## How it works

1. **Fetch.** If the input is a URL, `yt-dlp` grabs the best audio-only stream to a temp directory. If it's a local path, this step is skipped.
2. **Decode.** `ffmpeg` pipes the media into 16 kHz mono PCM. The optional `atempo` filter speeds it up without raising pitch.
3. **Recognize.** A streaming Zipformer2 transducer (Kroko English, INT8 ONNX, ~80 MB) eats the PCM in chunks. CPU-only. Greedy decode.
4. **Format.** Plain text by default. With `--timestamps`, token timestamps are grouped on `.!?` into sentences and scaled back to original-audio time.

With `--diarize`, a second pass runs the audio (decoded at original speed) through a speaker-segmentation model and a speaker-embedding model, clusters the voiceprints into speakers, and tags each sentence with the speaker active at its start. All ONNX, all CPU.

No frame is sent anywhere. No state is kept between runs except the cached model.

---

## Performance

yapsnap tunes itself to your CPU — you shouldn't need to touch any flags.

On first decode it detects your **physical** core count (not the logical/hyperthread count, which oversubscribes and runs *slower*) and splits the work into a few parallel decode processes that, together, use about one thread per physical core. On a 4-core laptop that's two workers of two threads each; on an 8-core machine, four workers of two threads. Short clips skip the split entirely and decode as a single stream, since chunking only pays off once there's enough audio to outweigh its per-chunk warmup.

Two knobs let you override the autotuning if you want to experiment:

- `--workers N` — number of parallel decode processes (`0` = autodetect, `1` = single stream, no chunking).
- `--threads N` — ONNX threads per worker (`0` = autodetect). Keeping `workers × threads` at or below your physical core count is the sweet spot; going above it tends to slow things down rather than speed them up.

`--speed` is the other lever: it shortens the audio before decoding, so higher values mean a faster run (at some accuracy cost on hard audio). The default `1.4` balances speed and readability.

`YAPSNAP_THREADS` overrides the detected core budget for a run if autodetection guesses wrong on an unusual machine.

---

## Model & cache

The default Kroko English model is downloaded on first run to:

- **macOS** — `~/Library/Caches/yapsnap/`
- **Linux** — `$XDG_CACHE_HOME/yapsnap/` (or `~/.cache/yapsnap/`)
- **Windows** — `%LOCALAPPDATA%\yapsnap\`

To use a different streaming transducer (other languages, larger Kroko variants, etc.), point `--model` at a directory containing `encoder(.int8).onnx`, `decoder(.int8).onnx`, `joiner(.int8).onnx`, and `tokens.txt`. Or set `KROKO_MODEL` in your environment.

If you use `--diarize`, the segmentation and embedding models download to a `diarization-models/` subfolder of the same cache directory on first use, and are reused offline thereafter.

---

## Other languages

The default model is English, but yapsnap isn't limited to it. To transcribe another language, just download the matching model and point yapsnap at it — no code changes, no reinstall.

Kroko publishes streaming models for a growing list of languages on Hugging Face: <https://huggingface.co/Banafo/Kroko-ASR/tree/main>. As of now that includes:

- Dutch
- French
- German
- Hebrew
- Italian
- Portuguese
- Spanish
- Swedish
- Swiss German
- Turkish

Download the one you need, unpack it into its own folder, and run:

```bash
# Per-run: pass the model folder explicitly
yapsnap interview.mp3 --model /path/to/kroko-french

# Or set it once as your default for the session
export KROKO_MODEL=/path/to/kroko-french
yapsnap interview.mp3
```

Each model is single-language, so to work across several languages keep them in separate folders and switch with `--model` (or re-export `KROKO_MODEL`) as you go. Any other sherpa-onnx streaming transducer with the standard `encoder` / `decoder` / `joiner` / `tokens.txt` layout works too, not just the Kroko ones.

---

## Diarization

`--diarize` adds speaker labels to the transcript — "who spoke when" — so each line is prefixed with `SPEAKER_00`, `SPEAKER_01`, and so on:

```bash
yapsnap interview.mp3 --diarize
```

```
SPEAKER_00 [00:00]: Welcome to the show.
SPEAKER_01 [00:03]: Glad to be here, thanks for having me.
SPEAKER_00 [00:08]: Let's get started.
```

It stays true to yapsnap's design: CPU-only, ONNX, no PyTorch, no extra runtime dependencies beyond the `sherpa-onnx` you already have. Two small models download once on first use (a speaker-segmentation model plus a speaker-embedding model) and cache alongside the ASR model.

### How the labels are produced

`--diarize` implies `--timestamps` — the two share a clock. Transcription runs on the sped-up audio as usual, while diarization runs on the same source decoded at original speed (`1.0x`), because speeding audio up degrades both speaker-boundary detection and the voiceprint embeddings. Each transcript sentence is then matched to whichever speaker was active at its start time.

Because diarization needs sentence timestamps to attach labels to, `--diarize` will stop with an error if your `sherpa-onnx` build doesn't expose timestamp data, rather than silently dropping the speaker labels.

### Speaker count

By default the number of speakers is detected automatically. Auto-detection is solid up to about seven speakers and degrades above that. If you know the count, pass it — it's more reliable:

```bash
yapsnap call.mp3 --diarize --num-speakers 2
```

### Choosing a segmentation model

| Model        | `--diarize-model` | License        | Notes                                          |
|--------------|-------------------|----------------|------------------------------------------------|
| pyannote 3.0 | `pyannote` (default) | CC-BY-4.0   | Attribution only; the safe default.            |
| Reverb v1    | `reverb`          | **Non-commercial** | Same architecture, fine-tuned for accuracy. |

```bash
yapsnap panel.mp4 --diarize --diarize-model reverb
```

`pyannote` is the default because its license is clean for most uses. `reverb` (Rev's fine-tune of the same architecture) can be more accurate but is distributed under a **non-commercial** license — yapsnap prints a reminder the first time you download it. Check the Rev model card before using it for anything commercial.

### Limits

- **No overlapping speech.** Each moment is assigned to exactly one speaker; simultaneous talking isn't modeled.
- **Speaker counting weakens past ~7 speakers.** Pass `--num-speakers` when you know it.
- **Labels are per-run.** `SPEAKER_00` is not the same person across different files.

To override the embedding model (for example if the default asset name ever changes), set `YAPSNAP_EMBEDDING_MODEL` to a different filename from the sherpa-onnx speaker-recognition release.

---

## Notes & limits

- The default model is **English**. For other languages, download a matching model and pass it with `--model` — see [Other languages](#other-languages) for the current list and instructions.
- `--speed` trades time-stretching for runtime: higher means less audio to decode and a shorter run (try `2.0` to go faster), lower means cleaner output on noisy, mumbled, or fast-speech sources (drop to `1.0`). The default `1.4` is a middle ground that reads well; the difference between nearby values like 1.4 and 1.5 is small, but across the full range (1.0 vs 2.0) it's noticeable.
- Some social-media URLs are geo-locked or login-walled; `yt-dlp` will say so explicitly.
- This is a streaming model, so timestamps come from token positions in the recognized stream. They're accurate enough for navigation, not for subtitling-grade alignment.

---

## License

Apache-2.0 for this project. The Kroko model is distributed under its own license — see <https://huggingface.co/Banafo/Kroko-ASR>. Powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).

The optional diarization models carry their own licenses, separate from yapsnap's: the default **pyannote** segmentation model is CC-BY-4.0 (attribution), the speaker-embedding model is Apache-2.0, and the opt-in **reverb** segmentation model (`--diarize-model reverb`) is **non-commercial**. If you use diarization, review the license of the model you select before relying on it.