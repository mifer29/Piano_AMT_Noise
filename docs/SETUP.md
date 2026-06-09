# Setup

[← Back to README](../README.md)

## Requirements

- **Python 3.10+**
- Python packages: see [requirements.txt](../requirements.txt) (PyTorch, librosa, soundfile,
  pretty_midi, h5py, pandas, numpy, scipy, pyroomacoustics, mir_eval, partitura, FastAPI,
  Uvicorn, OpenCV, matplotlib, …)
- System binaries on `PATH`: `ffmpeg`, `ffprobe`, and `fluidsynth` (for piano-roll video
  audio; the server degrades to silent video if missing)
- A General MIDI soundfont (the project uses a Yamaha C5 Grand)

## Create a virtual environment

```bash
# From the repository root, with Python 3.10+.
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate # macOS / Linux
# .venv\Scripts\activate  # Windows (PowerShell)

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The full install (PyTorch included) needs **~3 GB** of free disk space.

> `mpteval` (multi-pitch evaluation) may not be on PyPI: if `pip install` skips it,
> install it from source. It is only needed for [evaluation/](../evaluation/).

Install the system binaries separately (they are not pip packages):

```bash
# macOS (Homebrew)
brew install ffmpeg fluid-synth

# Debian / Ubuntu
sudo apt-get install ffmpeg fluidsynth
```

Deactivate the environment with `deactivate` when finished.

## Environment variables (`.env`)

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
