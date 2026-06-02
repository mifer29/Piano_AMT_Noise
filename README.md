# Piano_AMT_Noise

**Transformer-based Automatic Music Transcription for Piano with Noise Robustness and Mobile Deployment.**

Bachelor's Thesis (TFG)
Data Science and Engineering, Universidad Carlos III de Madrid (UC3M).
Author: Miguel Fernández Lara.

---

## Overview

The system proposed transcribes polyphonic piano audio into MIDI using a sequence-to-sequence
Transformer with a convolutional front-end. The input to the encoder are log-mel spectrograms and the decoder autoregressively emits a stream of MIDI-like event tokens (time / velocity / note-on / note-off).

The central research contribution is **demonstrated robustness under simulated acoustic
degradation**. The model is trained on MAESTRO with on-the-fly noise augmentation (drawn from the MUSAN corpus and simulated room/phone effects) and is evaluated across multiple controlled acoustic conditions. A FastAPI backend and an Android app (Jetpack Compose) wrap the trained model into a usable end-to-end transcription tool.

### Model variants

Three variants are compared in the experiments chapter:

| Variant | Architecture | Training data |
|---------|--------------|---------------|
| **M1**  | Baseline, no CNN front-end | Clean |
| **M9**  | Full architecture (CNN front-end + Transformer) | Clean |
| **M11** | Full architecture | Noise-augmented |

---

## Architecture

Defined in [model.py](noise_robust_model/model.py) (`AudioTransformer`):

- **CNN front-end**: a stem plus three residual conv blocks (GroupNorm, ReLU,
  1×1 conv shortcuts). Two frequency-stride-2 blocks reduce the mel axis by 4× before a
  linear projection to `d_model`, turning `[B, T, n_mels]` into `[B, T, d_model]`.
- **Encoder**: absolute sinusoidal positional encoding + pre-norm
  `TransformerEncoder` (8 layers, 8 heads, `d_model=512`, `d_ff=1024`).
- **Decoder**: token embedding + pre-norm `TransformerDecoder` with a causal mask and a
  padding mask (PAD = 0), projecting to the vocabulary.
- GPT-2-style residual scaling on initialization (`scale_residual_init`) to keep variance
  stable with depth.

Training uses teacher forcing (`shift_tokens_right`); inference uses greedy decoding
(`greedy_decode`).

### Tokenization

`build_vocab()` (see [server/vocab.py](server/vocab.py)) produces a **1387-token**
vocabulary:

- 3 special tokens: `PAD`, `BOS`, `EOS`
- 1000 `time_*` tokens (10 ms resolution)
- 128 `velocity_*` tokens
- 128 `note_on_*` + 128 `note_off_*` tokens

### Audio front-end parameters

`SAMPLE_RATE=16000`, `HOP_LENGTH=128`, `N_MELS=512`, `n_fft=2048`. Inference operates on
512-frame windows with **25 % overlap** (`STRIDE=384`); boundary duplicates are merged by a
post-hoc deduplication step (overlap-skip), so the deployed pipeline matches the one that
produced the reported metrics.

---

## Repository layout

```
.
├── aux/                    Data preparation pipeline
│   ├── preprocess_maestro.py        MIDI → note-event arrays (sustain extension, overlap fix)
│   ├── build_hdf5.py                Pack mels + events into HDF5
│   ├── build_testnoise.py           Build the deterministic 9-condition noisy test set
│   ├── build_testnoise.py / build_hdf5_testnoise.py
│   ├── precompute_mels_testnoise.py
│   └── preprocess_testnoise.py
│
├── baseline_model/         M1 — baseline (no CNN front-end): model.py, train.py, dataset
├── clean_model/            M9 — full architecture, clean-trained
├── noise_robust_model/     M11 — full architecture, noise-augmented training
│   └── (each: model.py, train.py, maestro_dataset.py)
│
├── data/                   EDA figures for the thesis (eda_figures.py)
├── debug/                  Diagnostic scripts (token imbalance, mel/noise checks, overfit-batch)
│
├── evaluation/
│   ├── evaluate.py                  Clean MAESTRO test-set eval (mir_eval + mpteval)
│   ├── evaluate_testnoise.py        Eval across the 9 noisy conditions
│   └── transcribe.py                Standalone WAV → MIDI CLI
│
├── results/                Saved metrics
│   ├── ablation/                    M1–M11 ablation runs
│   ├── clean_test_eval/             Clean test metrics
│   └── noise_test_eval/             Per-condition noisy test metrics (M1/M9/M11)
│
├── server/                 FastAPI inference backend
│   ├── server.py                    API, startup checks, overlap-skip inference
│   ├── config.py                    Paths, audio + video config, request limits
│   ├── audio.py                     Mel, token→note conversion, deduplication
│   ├── midi_utils.py                Notes → MIDI, tempo estimation
│   ├── video.py                     Piano-roll video rendering (OpenCV + FluidSynth)
│   └── vocab.py                     Token vocabulary (must match training)
│
└── app/                    Android app (Kotlin / Jetpack Compose, es.uc3m.android.pianotranscriber)
```

---

## The evaluation conditions

Built deterministically (seed 42) in [aux/build_testnoise.py](aux/build_testnoise.py):

| Condition | Description |
|-----------|-------------|
| `clean` | Unmodified reference |
| `noise_20db` | MUSAN noise @ 20 dB SNR (within training range) |
| `noise_10db` | MUSAN noise @ 10 dB SNR (below training range) |
| `noise_5db` | MUSAN noise @ 5 dB SNR (severe; degradation-curve tail) |
| `speech_15db` | MUSAN speech interference @ 15 dB SNR |
| `phone_eq` | Phone-mic coloration (300 Hz high-pass + presence boost) |
| `reverb_small` | Small domestic room (pyroomacoustics, in-distribution) |
| `reverb_large` | Larger, less absorptive room (out-of-distribution) |
| `phone_simulation` | Phone EQ → small-room reverb → noise @ 15 dB chain |

Training augmentation includes room reverb (pyroomacoustics), a 100 Hz Butterworth
high-pass, MUSAN background/speech mixing (SNR 15–35 dB), and AAC codec compression. It does
**not** include distance low-pass filtering or AGC. Robustness claims are scoped to
*simulated acoustic degradation*, not real-world phone recordings.

---

## Setup

### Requirements

- **Python 3.10+**
- Python packages — see [requirements.txt](requirements.txt) (PyTorch, librosa, soundfile,
  pretty_midi, h5py, pandas, numpy, scipy, pyroomacoustics, mir_eval, partitura, FastAPI,
  Uvicorn, OpenCV, matplotlib, …)
- System binaries on `PATH`: `ffmpeg`, `ffprobe`, and `fluidsynth` (for piano-roll video
  audio; the server degrades to silent video if missing)
- A General MIDI soundfont (the project uses a Yamaha C5 Grand)

### Create a virtual environment

```bash
# From the repository rootuse,  Python 3.10+.
# The pinned versions require Python >= 3.10; the system Python 3.9 will fail with "No matching distribution".
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate # macOS / Linux
# .venv\Scripts\activate  # Windows (PowerShell)

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The full install (PyTorch included) needs **~3 GB** of free disk space.

> `mpteval` (multi-pitch evaluation) may not be on PyPI — if `pip install` skips it,
> install it from source. It is only needed for [evaluation/](evaluation/).

Install the system binaries separately (they are not pip packages):

```bash
# macOS (Homebrew)
brew install ffmpeg fluid-synth

# Debian / Ubuntu
sudo apt-get install ffmpeg fluidsynth
```

Deactivate the environment with `deactivate` when finished.

### Environment variables (`.env`)

```dotenv
# Data preprocessing
maestro_root=/path/to/maestro-v3.0.0
events_dir=/path/to/note_events
mel_dir=/path/to/mels
split=train|validation|test
output_path=/path/to/output
hdf5_path=/path/to/dataset.h5

# Inference / server
model_path=/path/to/checkpoint.pt
soundfont_path=/path/to/YamahaC5Grand.sf2
```

---

## Usage

### 1. Data preparation

```bash
python aux/preprocess_maestro.py        # MIDI to note-event arrays
python aux/build_hdf5.py                # pack mels + events into HDF5
python aux/build_testnoise.py           # build the conditions noisy test set
python aux/build_hdf5_testnoise.py      # pack the noisy test set
```

### 2. Training

Each variant has its own `train.py`. Example (M11, noise-robust):

```bash
python noise_robust_model/train.py \
    --hdf5_train  /path/to/train.h5 \
    --hdf5_val    /path/to/val.h5 \
    --output_root runs/ \
    --batch_size  256 \
    --total_steps 600000
```

Runs are written to a timestamped folder with checkpoints, `train.log`, and `metrics.pt`.
Use `--resume_run <dir>` to continue.

### 3. Evaluation

```bash
# Clean MAESTRO test set
python evaluation/evaluate.py

# Across the noisy conditions
python evaluation/evaluate_testnoise.py
```

Metrics are computed with `mir_eval` (onset/offset/velocity F1) and `mpteval`, and saved
under [results/](results/).

### 4. Single-file transcription

```bash
python evaluation/transcribe.py  input.wav  output.mid
```

### 5. Inference server

```bash
cd server
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server runs startup checks (binaries, checkpoint, soundfont), then accepts audio
uploads (≤ 300 MB, ≤ 300 s), runs overlap-skip windowed inference, and returns MIDI plus an
optional piano-roll video. Designed to run on a laptop CPU.

### 6. Android app

Open [app/](app/) in Android Studio and build the
`es.uc3m.android.pianotranscriber` module. The app records or selects audio, uploads it to
the backend, and renders the returned transcription as a piano roll.

---

## Notes & scope

- **Sim-to-real gap is documented.** The model generalizes across piano timbres
  and recording conditions *within a defined operating envelope*; home-phone-recording
  limitations are a stated scope boundary.
- **Overlap-skip by construction.** Duplicate boundary notes are handled by the inference
  design, not patched after the fact.
- Velocity is quantized to 128 bins; time resolution is 10 ms.

---

*Universidad Carlos III de Madrid. Bachelor's Thesis in Data Science and Engineering, 2026.*
