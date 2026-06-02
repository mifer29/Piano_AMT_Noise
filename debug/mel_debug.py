import numpy as np
import torch
import random
import librosa
import soundfile as sf
from pathlib import Path
from maestro_dataset import MAESTROSeq2SeqDataset
from dotenv import load_dotenv
import os
load_dotenv()

# ==========================================================
# CONFIG
# ==========================================================

maestro_root = os.getenv("maestro_root")
events_dir = os.getenv("events_dir")
mel_dir = os.getenv("mel_dir")

TEST_DATASET_SLICE = True
TEST_ON_THE_FLY_MEL = True
TEST_FULL_PIPELINE = True

SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 128
N_MELS = 512

FLOAT16_TOLERANCE = 1e-2  # float16 max rounding error

random.seed(42)

# ==========================================================
# Create dataset
# ==========================================================

dataset = MAESTROSeq2SeqDataset(
    maestro_path=maestro_root,
    events_dir=events_dir,
    mel_dir=mel_dir,
    split="train"
)

print(f"Dataset size: {len(dataset)}")

idx = 0
file_info = dataset.file_list[idx]
mel_path = file_info["mel_path"]
n_frames = file_info["n_frames"]

max_start_frame = max(0, n_frames - dataset.max_input_frames)
start_frame = random.randint(0, max_start_frame)

print(f"Start frame used: {start_frame}")
print(f"Mel path: {mel_path}")

# ==========================================================
# Raw npy sanity check — do this before anything else
# ==========================================================

print("\n=== RAW NPY SANITY CHECK ===")
raw = np.load(mel_path, mmap_mode="r")
print(f"Stored mel dtype:  {raw.dtype}")           # expect float16
print(f"Stored mel shape:  {raw.shape}")            # expect [n_frames, 512]
print(f"Stored mel min:    {raw.min():.4f}")        # log scale, expect ~ -14 to 0
print(f"Stored mel max:    {raw.max():.4f}")
print(f"Stored mel mean:   {raw.mean():.4f}")
print(f"Has NaN:           {np.isnan(raw).any()}")
print(f"Has Inf:           {np.isinf(raw).any()}")
del raw

# ==========================================================
# Load dataset segment — keep as float32 (what model sees)
# ==========================================================

# _load_mel_segment returns float32 (upcasted from float16 on disk)
dataset_segment = torch.from_numpy(
    dataset._load_mel_segment(mel_path, start_frame)
)  # float32

print("\n=== DATASET SEGMENT INFO (float32, what model sees) ===")
print(f"Shape: {dataset_segment.shape}")
print(f"Dtype: {dataset_segment.dtype}")
print(f"Min:   {dataset_segment.min().item():.4f}")
print(f"Max:   {dataset_segment.max().item():.4f}")
print(f"Mean:  {dataset_segment.mean().item():.4f}")
print(f"Std:   {dataset_segment.std().item():.4f}")
print(f"Has NaN: {torch.isnan(dataset_segment).any().item()}")


# ==========================================================
# TEST 1: Dataset slice vs raw npy slice
# ==========================================================

def test_dataset_slice():
    print("\n==============================")
    print("TEST 1: DATASET VS RAW SLICE")
    print("==============================")

    original_mel = np.load(mel_path)  # float16 on disk
    print(f"Raw npy dtype: {original_mel.dtype}")

    expected_segment = original_mel[
        start_frame:start_frame + dataset.max_input_frames
    ]

    if expected_segment.shape[0] < dataset.max_input_frames:
        pad = np.zeros(
            (dataset.max_input_frames - expected_segment.shape[0], dataset.n_mels),
            dtype=np.float16
        )
        expected_segment = np.vstack([expected_segment, pad])

    # Convert to float32 for comparison — same as what _load_mel_segment does
    expected_segment = torch.tensor(expected_segment, dtype=torch.float32)

    difference = torch.abs(expected_segment - dataset_segment)

    print(f"Max abs difference:  {difference.max().item():.6f}")
    print(f"Mean abs difference: {difference.mean().item():.6f}")
    print(f"Exact equality:      {torch.equal(expected_segment, dataset_segment)}")

    if difference.max().item() == 0:
        print("\n✅ Dataset slicing is PERFECT.")
    else:
        print("\n❌ Dataset slicing mismatch.")


# ==========================================================
# TEST 2: Precomputed mel vs on-the-fly mel
# ==========================================================

def test_on_the_fly_mel():
    print("\n====================================")
    print("TEST 2: PRECOMPUTED VS ON-THE-FLY")
    print("====================================")

    mel_stem = Path(mel_path).stem.replace("_mel", "")

    for _, row in dataset.metadata.iterrows():
        if Path(row["audio_filename"]).stem == mel_stem:
            audio_rel_path = row["audio_filename"]
            break
    else:
        raise ValueError("Audio file not found in metadata")

    audio_path = dataset.maestro_path / audio_rel_path
    print(f"Audio path: {audio_path}")

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        n_mels=N_MELS,
        power=2.0,
    )
    mel = np.log(mel + 1e-6)
    mel = mel.T  # [n_frames, n_mels]

    expected_segment = mel[start_frame:start_frame + dataset.max_input_frames]

    if expected_segment.shape[0] < dataset.max_input_frames:
        pad = np.zeros(
            (dataset.max_input_frames - expected_segment.shape[0], N_MELS),
            dtype=np.float32
        )
        expected_segment = np.vstack([expected_segment, pad])

    # Keep as float32 — do NOT cast to float16 here
    # float16 truncation in preprocessing is expected to introduce small errors
    expected_segment = torch.tensor(expected_segment, dtype=torch.float32)

    difference = torch.abs(expected_segment - dataset_segment)

    print(f"Max abs difference:  {difference.max().item():.6f}")
    print(f"Mean abs difference: {difference.mean().item():.6f}")
    print(f"Exact equality:      {torch.equal(expected_segment, dataset_segment)}")

    # Not exact due to float16 truncation in preprocessing — use tolerance
    if difference.mean().item() < FLOAT16_TOLERANCE:
        print(f"\n✅ Mel pipeline matches within float16 tolerance ({FLOAT16_TOLERANCE}).")
    else:
        print(f"\n❌ Mel preprocessing mismatch detected (mean diff > {FLOAT16_TOLERANCE}).")


# ==========================================================
# TEST 3: Full dataset pipeline (__getitem__)
# ==========================================================

def test_full_pipeline():
    print("\n====================================")
    print("TEST 3: FULL DATASET PIPELINE")
    print("====================================")

    # Seed BEFORE dataset[idx] so we can recover the same start_frame after
    random.seed(42)
    sample = dataset[idx]

    # Immediately re-seed and replay the same randint to get start_frame
    random.seed(42)
    file_info = dataset.file_list[idx]
    n_frames = file_info["n_frames"]
    max_start_frame = max(0, n_frames - dataset.max_input_frames)
    recovered_start_frame = random.randint(0, max_start_frame)

    print(f"Recovered start_frame: {recovered_start_frame}")

    # float32 — what the model actually receives
    pipeline_segment = sample["audio_features"]
    print(f"Segment shape: {pipeline_segment.shape}")
    print(f"Segment dtype: {pipeline_segment.dtype}")

    # Recompute expected from raw npy using the recovered start_frame
    expected = torch.from_numpy(
        dataset._load_mel_segment(mel_path, recovered_start_frame)
    )  # float32

    difference = torch.abs(expected - pipeline_segment)

    print(f"Max abs difference:  {difference.max().item():.6f}")
    print(f"Mean abs difference: {difference.mean().item():.6f}")
    print(f"Exact equality:      {torch.equal(expected, pipeline_segment)}")

    if difference.max().item() == 0:
        print("\n✅ FULL DATA PIPELINE IS PERFECT.")
    elif difference.mean().item() < FLOAT16_TOLERANCE:
        print(f"\n✅ Matches within float16 tolerance ({FLOAT16_TOLERANCE}).")
    else:
        print("\n❌ FULL PIPELINE MISMATCH DETECTED.")


# ==========================================================
# RUN TESTS
# ==========================================================

if TEST_DATASET_SLICE:
    test_dataset_slice()

if TEST_ON_THE_FLY_MEL:
    test_on_the_fly_mel()

if TEST_FULL_PIPELINE:
    test_full_pipeline()