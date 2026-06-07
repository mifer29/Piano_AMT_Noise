# Inference server & Android app

[← Back to README](../README.md) · See also [USAGE.md](USAGE.md), [SETUP.md](SETUP.md) and [STRUCTURE.md](STRUCTURE.md).

## 1. Inference server

Download M11.pt (final noise-robust model) from zenodo: [doi.org/10.5281/zenodo.20473425](https://doi.org/10.5281/zenodo.20473425).

Download the soundfont: [https://musical-artifacts.com/artifacts/4225](https://musical-artifacts.com/artifacts/4225)

Configure the correct path for the model M11.pt and soundfont in the .env

Now, to activate the server:

```bash
cd server
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server runs startup checks (binaries, checkpoint, soundfont), then accepts audio
uploads (≤ 300 MB, ≤ 300 s), runs overlap-skip windowed inference, and returns MIDI plus an
optional piano-roll video. Designed to run on a laptop CPU.

## 2. Android app

Option A: Open [app/](../app/) in Android Studio and build the
`es.uc3m.android.pianotranscriber` module. The app records or selects audio, uploads it to
the backend, and renders the returned transcription as a piano roll.

Option B: Download the .apk file and install it directly on a mobile device
