# Usage

[← Back to README](../README.md) · See also [SETUP.md](SETUP.md), [DEPLOYMENT.md](DEPLOYMENT.md) and [STRUCTURE.md](STRUCTURE.md).

## 1. Data preparation

```bash
python aux/preprocess_maestro.py        # MIDI to note-event arrays
python aux/build_hdf5.py                # pack mels + events into HDF5
python aux/build_testnoise.py           # build the conditions noisy test set
python aux/build_hdf5_testnoise.py      # pack the noisy test set
```

## 2. Training

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

### Hardware & training environment

The models in this thesis were trained on the **UC3M DTSC Hypercomputing Cluster** (Mesos-based
batch scheduler), on **NVIDIA RTX 4090 (24 GB)** GPUs: the full output vocabulary and
feed-forward width make 24 GB the practical minimum to train without memory overflow. Jobs
requested 2 GPUs, 12 CPU cores, and 64 GB RAM, inside an isolated Conda environment with CUDA on
the path; `OMP_NUM_THREADS=MKL_NUM_THREADS=1` to avoid CPU oversubscription, and four persistent
DataLoader workers reading sample chunks directly from the HDF5 container by key (prefetch +
pinned memory).

The training script is **scheduler-agnostic**: it is a plain `python train.py` invocation that
runs unchanged on a single GPU or any other cluster. The effective batch size of 256 is reached
by **gradient accumulation** (16 micro-batches), not data parallelism, so only the *micro*-batch
size is GPU-memory-bound; adjust `--batch_size` to your VRAM and `--total_steps` to your budget.
When more than one GPU is visible, the model is additionally wrapped in `DataParallel`, but the
schedule is always defined in terms of micro-batch steps.

| | M1 | M9 | **M11** |
|---|:---:|:---:|:---:|
| Parameters (M) | 48.7 | 51.6 | 52.2 |
| Micro-batch size | 32 | 16 | 16 |
| Grad-accumulation steps | — | 16 | 16 |
| Effective batch size | 32 | 256 | 256 |
| Micro-batch steps | 400k | 400k | 600k |
| Wall-clock (RTX 4090) | 19 h | 12 h | 60 h |

> **Storage.** Reproducing the full pipeline needs **~600 GB**: MAESTRO v3.0.0 (120 GB) and MUSAN
> (7.2 GB) raw, precomputed mels (~165 GB NPY), the HDF5 train/val/test containers (clean ~86 GB,
> noise-augmented ~162 GB), and checkpoints (~600 MB each). Mel pre-computation + HDF5 packing
> trade disk for training throughput by removing per-sample file I/O and `librosa` work from the
> training loop. See Appendix A of the thesis for the full breakdown.

## 3. Evaluation

```bash
# Clean MAESTRO test set
python evaluation/evaluate.py

# Across the noisy conditions
python evaluation/evaluate_testnoise.py
```

Metrics are computed with `mir_eval` (onset/offset/velocity F1) and `mpteval` (musically-informed metrics), and saved
under [results/](../results/).

## 4. Single-file transcription

```bash
python evaluation/transcribe.py  input.wav  output.mid
```

## 5. Inference server & Android app

To run the FastAPI inference server and build or install the Android app, see
[DEPLOYMENT.md](DEPLOYMENT.md).

