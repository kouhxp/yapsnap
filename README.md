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
- 🪶 **One file, three deps.** `sherpa-onnx`, `numpy`, `yt-dlp`. The whole tool is a single Python module.
- 🗣 **Ten-plus languages.** English out of the box; French, German, Spanish, Italian, Portuguese, Dutch, Swedish, Swiss German, Hebrew, and Turkish are a one-line `--model` swap away. See [Other languages](#other-languages).
- ⏱ **Sentence-level timestamps when you want them.** `--timestamps` adds `[MM:SS]` per sentence using Kroko's built-in punctuation. Timing stays correct even when you transcribe at 2x.

---

## Quickstart

```bash
# 1. ffmpeg on PATH (one-time, per OS — see below)
# 2. Install
pip install .

# 3. Snap something
yapsnap https://www.tiktok.com/@user/video/7234567890123456789
yapsnap meeting.mp4 --timestamps
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

```bash
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

# Custom output path
yapsnap input.mp4 -o ./transcripts/talk.txt

# Don't speed audio up before transcribing (default is 1.5x, pitch preserved)
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

Timestamps stay in original-audio time even at `--speed 1.5` or higher.

---

## Flags

| Flag              | Description                                                          |
|-------------------|----------------------------------------------------------------------|
| `-o`, `--output`  | Output `.txt` path. Default: `./transcripts/<input>_transcript.txt`. |
| `--timestamps`    | Emit `[MM:SS] sentence.` lines instead of a single paragraph.        |
| `--speed`         | Pre-transcription speedup factor, pitch preserved. Default `1.5`.    |
| `--keep-audio`    | Keep the downloaded audio (URL inputs only).                         |
| `--model`         | Override the model directory. Also reads `KROKO_MODEL` env var.      |

---

## How it works

1. **Fetch.** If the input is a URL, `yt-dlp` grabs the best audio-only stream to a temp directory. If it's a local path, this step is skipped.
2. **Decode.** `ffmpeg` pipes the media into 16 kHz mono PCM. The optional `atempo` filter speeds it up without raising pitch.
3. **Recognize.** A streaming Zipformer2 transducer (Kroko English, INT8 ONNX, ~80 MB) eats the PCM in chunks. CPU-only. Greedy decode.
4. **Format.** Plain text by default. With `--timestamps`, token timestamps are grouped on `.!?` into sentences and scaled back to original-audio time.

No frame is sent anywhere. No state is kept between runs except the cached model.

---

## Model & cache

The default Kroko English model is downloaded on first run to:

- **macOS** — `~/Library/Caches/yapsnap/`
- **Linux** — `$XDG_CACHE_HOME/yapsnap/` (or `~/.cache/yapsnap/`)
- **Windows** — `%LOCALAPPDATA%\yapsnap\`

To use a different streaming transducer (other languages, larger Kroko variants, etc.), point `--model` at a directory containing `encoder(.int8).onnx`, `decoder(.int8).onnx`, `joiner(.int8).onnx`, and `tokens.txt`. Or set `KROKO_MODEL` in your environment.

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

## Notes & limits

- The default model is **English**. For other languages, download a matching model and pass it with `--model` — see [Other languages](#other-languages) for the current list and instructions.
- `--speed 1.5` shaves about a third off transcription time with minimal accuracy cost. Try `2.0` if you want it even faster, or `1.0` for noisy, mumbled, or fast-speech sources.
- Some social-media URLs are geo-locked or login-walled; `yt-dlp` will say so explicitly.
- This is a streaming model, so timestamps come from token positions in the recognized stream. They're accurate enough for navigation, not for subtitling-grade alignment.

---

## License

Apache-2.0 for this project. The Kroko model is distributed under its own license — see <https://huggingface.co/Banafo/Kroko-ASR>. Powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [yt-dlp](https://github.com/yt-dlp/yt-dlp).