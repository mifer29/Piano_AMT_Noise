# Repository layout

[← Back to README](../README.md)

```
.
├── aux/                    Data preparation pipeline
│   ├── preprocess_maestro.py        MIDI to note-event arrays (sustain extension, overlap fix)
│   ├── build_hdf5.py                Pack mels + events into HDF5
│   ├── build_testnoise.py           Build the deterministic 9-condition noisy test set
│   ├── build_hdf5_testnoise.py      Pack the noisy test set into HDF5
│   ├── precompute_mels_testnoise.py Precompute mels for the noisy conditions
│   └── preprocess_testnoise.py      Note-event arrays for the noisy test set
│
├── baseline_model/         M1: baseline (no CNN front-end): model.py, train.py, dataset
├── clean_model/            M9: full architecture, clean-trained
├── noise_robust_model/     M11: full architecture, noise-augmented training
│   └── (each: model.py, train.py, maestro_dataset.py)
│
├── data/                   EDA MAESTRO dataset
├── debug/                  Diagnostic scripts (token imbalance, mel/noise checks, overfit-batch)
│
├── evaluation/
│   ├── evaluate.py                  Clean MAESTRO test-set eval (mir_eval + mpteval)
│   ├── evaluate_testnoise.py        Eval across the noisy conditions
│   └── transcribe.py                Standalone WAV to MIDI CLI
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
│   ├── midi_utils.py                Notes to MIDI, tempo estimation
│   ├── video.py                     Piano-roll video rendering (OpenCV + FluidSynth)
│   └── vocab.py                     Token vocabulary (must match training)
│
└── app/                    Android app (Kotlin / Jetpack Compose, es.uc3m.android.pianotranscriber)
```
