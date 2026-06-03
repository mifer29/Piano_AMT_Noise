# **Transformer-based Automatic Music Transcription for Piano with Noise Robustness and Mobile Deployment.**

Bachelor's Thesis (TFG)<br>
Data Science and Engineering, Universidad Carlos III de Madrid (UC3M).<br>
Author: Miguel Fernández Lara.<br>
Tutor: Dr. Víctor P. Gil Jiménez.<br>
Madrid, June 2026.

---

## Abstract

This thesis presents the development and implementation of an end-to-end Automatic Music Transcription (AMT) system for polyphonic piano recordings captured under noisy conditions, where existing models tend to degrade. The proposed approach is a sequence-to-sequence Transformer with a convolutional front-end for feature extraction, which maps mel-spectrogram inputs to a token vocabulary encoding note onsets, offsets, and velocities. To address the gap between clean audio and realistic recording conditions, the model is trained to be robust to acoustic degradation through an augmentation pipeline that simulates mobile recording conditions, including room reverberation, environmental noise from the MUSAN dataset, and signal-processing effects. 

The system is trained on the MAESTRO dataset and evaluated on clean audio and a range of acoustic degradation conditions: varying levels of background and speech noise, reverberant environments, and simulated phone recordings. The noise-robust model substantially improves transcription quality under the degraded conditions compared to the same model trained without augmentation, while preserving transcription quality on clean audio.
    
Furthermore, the work includes the deployment of a client-server mobile application that allows users to record a piano performance from their phone and automatically receive a MIDI transcription and a Synthesia-style piano roll video. Together, the augmentation methodology, the sequence-to-sequence token formulation, and the deployment of the mobile system demonstrate that piano AMT models can be made robust to realistic acoustic degradation while remaining practical for end-user deployment.

**Keywords:** Automatic Music Transcription · Polyphonic Piano Transcription · Noise
Robustness · Deep Learning · Transformers · Convolutional Neural Networks · Sequence-to-Sequence Models · Data Augmentation · MAESTRO · MUSAN · Mobile Application

---

## Overview

This repository contains the code and results of the piano AMT system proposed in the thesis, which transcribes polyphonic piano audio into MIDI using a sequence-to-sequence Transformer with a convolutional front-end. 

The main research contribution is **demonstrated robustness under simulated acoustic degradation**. The model is trained on MAESTRO with on-the-fly noise augmentation (drawn from the MUSAN corpus and simulated room/phone effects) and is evaluated across multiple controlled acoustic conditions. A FastAPI backend and an Android app (Jetpack Compose) wrap the trained model into a usable end-to-end transcription tool.

> **Headline result.** Averaged across the eight degraded test conditions, noise-augmented
> training (M11) raises onset F1 from **76.4 % to 87.4 %** over the same architecture trained on
> clean audio (M9): a **+11.0-point** gain, widening to **+16.0 points** on the strictest
> onset + offset + velocity metric, while preserving clean-audio accuracy (97.1 %). The gains
> are largest under speech interference and severe noise (up to ~+30 points). See
> [Results](#results).

### Model variants

An ablation is performed evaluating performance of 11 different models. Three selected variants are compared in the experiments chapter:

| Variant | Architecture | Training data |
|---------|--------------|---------------|
| **M1**  | Baseline, no CNN front-end | Clean |
| **M9**  | Full architecture (CNN front-end + Transformer) | Clean |
| **M11** | Full architecture | Noise-augmented |

The trained checkpoints for these three variants (M1, M9, M11) are available on Zenodo:
[doi.org/10.5281/zenodo.20473425](https://doi.org/10.5281/zenodo.20473425).

---

## Noisy Test Set Results

Onset **F1 (%)** on the held-out MAESTRO test set, across the evaluation conditions
(full metrics: onset/offset/velocity, plus `mpteval` in [results/](results/)):

| Condition | M1 (baseline) | M9 (clean) | **M11 (noise-aug)** |
|-----------|:---:|:---:|:---:|
| `clean` | 96.4 | 96.8 | **97.1** |
| `noise_20db` | 90.8 | 89.8 | **95.2** |
| `noise_10db` | 76.4 | 76.5 | **91.0** |
| `noise_5db` | 55.8 | 61.5 | **87.4** |
| `speech_15db` | 62.4 | 63.8 | **93.0** |
| `phone_eq` | 73.7 | 93.0 | **94.2** |
| `reverb_small` | 93.0 | 88.9 | 83.7 |
| `reverb_large` | 59.5 | 67.8 | **73.7** |
| `phone_simulation` | 53.6 | 69.8 | **80.6** |

The noise-augmentated model (M11) improves robustness across every degraded condition and is most
decisive where the baseline collapses: `noise_5db` (+31.6), `speech_15db` (+30.6),
`phone_simulation` (+27.0), while leaving clean-audio accuracy essentially unchanged. The one regression is `reverb_small`, where the clean-trained models score higher; this in-distribution reverberation case is discussed in the experiments chapter.

### Onset + offset

Onset + offset **F1 (%)** (a note is correct only if both its onset *and* offset match):

| Condition | M1 (baseline) | M9 (clean) | **M11 (noise-aug)** |
|-----------|:---:|:---:|:---:|
| `clean` | 77.8 | 83.9 | **86.0** |
| `noise_20db` | 67.5 | 72.5 | **81.7** |
| `noise_10db` | 49.6 | 56.7 | **74.9** |
| `noise_5db` | 36.4 | 41.5 | **67.6** |
| `speech_15db` | 36.1 | 44.0 | **77.9** |
| `phone_eq` | 48.2 | 73.9 | **78.4** |
| `reverb_small` | 69.0 | 69.4 | 68.2 |
| `reverb_large` | 25.1 | 35.0 | **43.3** |
| `phone_simulation` | 27.6 | 46.0 | **56.7** |

### Onset + offset + velocity

Onset + offset + velocity **F1 (%)** (the strictest metric: onset, offset, *and* velocity must all match):

| Condition | M1 (baseline) | M9 (clean) | **M11 (noise-aug)** |
|-----------|:---:|:---:|:---:|
| `clean` | 76.5 | 81.9 | **85.0** |
| `noise_20db` | 62.9 | 66.7 | **80.5** |
| `noise_10db` | 44.9 | 50.9 | **73.2** |
| `noise_5db` | 32.1 | 36.0 | **64.3** |
| `speech_15db` | 33.4 | 40.9 | **76.8** |
| `phone_eq` | 45.3 | 70.7 | **76.0** |
| `reverb_small` | 67.1 | 65.6 | 66.2 |
| `reverb_large` | 22.9 | 31.1 | **40.3** |
| `phone_simulation` | 24.5 | 40.7 | **53.4** |

The offset and velocity metrics track the onset trend: M11 wins everywhere except the
in-distribution `reverb_small`, but the absolute scores are lower across the board, since
offsets blur under degradation faster than onsets do (the failure mode noted in
[Limitations](#limitations)). The largest noise-augmentation gains again land on speech and
severe noise: `speech_15db` (+41.8 offset, +43.4 offset+vel over M1) and `noise_5db`
(+31.2 / +32.2).

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

Training uses teacher forcing (`shift_tokens_right`); inference uses greedy decoding
(`greedy_decode`).

The input to the encoder are log-mel spectrograms and the decoder autoregressively emits a stream of MIDI-like event tokens (time / velocity / note-on / note-off).

### Tokenization

`build_vocab()` (see [server/vocab.py](server/vocab.py)) produces a **1387-token**
vocabulary:

- 3 special tokens: `PAD`, `BOS`, `EOS`
- 1000 `time_*` tokens (10 ms resolution)
- 128 `velocity_*` tokens
- 128 `note_on_*` + 128 `note_off_*` tokens

### Audio front-end parameters

`SAMPLE_RATE=16000`, `HOP_LENGTH=128`, `N_MELS=512`, `n_fft=2048`. Inference operates on
512-frame windows with **25 % overlap** (`STRIDE=384`); boundary duplicates are merged by a post-hoc deduplication step (overlap-skip), so the deployed pipeline matches the one that produced the reported metrics.

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
| `phone_simulation` | Phone EQ -> small-room reverb -> noise @ 15 dB chain |

Training augmentation includes room reverb (pyroomacoustics), a 100 Hz Butterworth
high-pass, MUSAN background/speech mixing (SNR 15–35 dB), and AAC codec compression. Robustness claims are scoped to
*simulated acoustic degradation*, not real-world phone recordings.

---

## Documentation

Install steps, commands, and the file tree live in [`docs/`](docs/) to keep this README focused
on the research:

- **[Setup](docs/SETUP.md)**: Requirements, virtual environment, system binaries, and `.env`.
- **[Usage](docs/USAGE.md)**: Data preparation, training (with the HPC / hardware notes),
  evaluation, single-file transcription, the inference server, and the Android app.
- **[Repository layout](docs/STRUCTURE.md)**: Annotated file tree.

---

## Limitations

Scoped honestly, per the thesis (Chapter 9):

- **Reverberation is the primary failure mode.** Onsets stay detectable, but offsets blur
  under room reflections: large-room O+Off F1 falls to ~43 %. Offset and velocity prediction degrade faster than onset detection across every condition.
- **Simulated, not real-world, degradation.** Robustness is measured on MUSAN-augmented
  MAESTRO with paired ground truth. Real phone recordings have no ground-truth MIDI, so no
  quantitative real-world claim is made; the sim-to-real gap is an open problem.
- **Piano only.** The vocabulary, 88-pitch range, and tokenization are piano-specific, and
  training uses only Yamaha Disklavier grand-piano timbres (MAESTRO). Other instruments,
  non-classical genres, and upright/digital pianos are out of scope.
- **Moderate musical expressiveness.** Note-level accuracy is strong, but dynamics are weaker: plausibly a cost of noise-augmented training.
- **Server-side inference.** The 52.2 M-parameter model runs on a laptop CPU; on-device
  deployment would require quantization/distillation and is left as future work.


---

## License

- **Thesis text and figures:** Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**).
- **Code in this repository:** **MIT** (see [LICENSE](LICENSE)). The permissive code license does not override the dataset terms below: trained weights and any redistributed data inherit them.
- **Datasets (not redistributed here):** MAESTRO v3.0.0 is **CC BY-NC-SA 4.0** and MUSAN is **CC BY 4.0**: both non-/share-alike terms carry over to models trained on them, so any commercial use would require separate clearance.

---

*Universidad Carlos III de Madrid. Bachelor's Thesis in Data Science and Engineering, 2026.*
