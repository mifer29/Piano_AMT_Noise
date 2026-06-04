"""
Precomputes mel spectrograms from the fixed test set produced by build_testnoise.py.

Input structure:
    test_set/
        clean/  song1.wav  song2.wav  ...
        noise_20db/ song1.wav  song2.wav  ...
        noise_10db/  ...
        noise_5db/ ...
        speech_15db/  ...
        reverb_large/ ...
        reverb_small/ ...
        phone_eq/  ...
        phone_simulation/ ...
        manifest.csv
        build_config.json

Output structure (mirrors input):
    test_mels/
        clean/ song1.npy  song2.npy...
        noise_20db/  song1.npy  song2.npy...
        ...

Each .npy file is a float16 array of shape (T, n_mels): identical format to the mels stored in the training HDF5, 
so the existing evaluation code can load them without modification
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm



# Worker function (must be top-level for multiprocessing pickling)


def compute_mel_for_file(task: tuple):
    """
    Loads one WAV file, computes its log-mel spectrogram, and saves it as .npy.

    task: (wav_path, npy_path, sample_rate, hop_length, n_fft, n_mels)
    """
    wav_path, npy_path, sample_rate, hop_length, n_fft, n_mels = task

    # Skip if already computed: allows resuming interrupted runs
    if npy_path.exists():
        return str(npy_path), "skipped"

    try:
        # Load: soundfile is faster than librosa for WAV
        audio, sr = sf.read(str(wav_path))

        # Mixdown to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        # Resample only if needed (build_testnoise already writes at target SR)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Log-mel spectrogram: identical parameters to training pipeline
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            power=2.0,
        )

        # Log compression with same floor as training HDF5
        mel = np.log(mel + 1e-6)

        # Transpose to (T, n_mels) and store as float16 to match HDF5 format
        mel = mel.T.astype(np.float16)

        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), mel)

        return str(npy_path), "ok"

    except Exception as e:
        return str(wav_path), f"ERROR: {e}"


def main(args):

    test_set_dir = Path(args.test_set_dir)
    output_dir   = Path(args.output_dir)

    if not test_set_dir.exists():
        raise FileNotFoundError(f"Test set directory not found: {test_set_dir}")

    # Discover condition folders: any subdirectory that contains WAV files
    condition_dirs = sorted([
        d for d in test_set_dir.iterdir()
        if d.is_dir() and any(d.glob("*.wav"))
    ])

    if not condition_dirs:
        raise FileNotFoundError(
            f"No condition subdirectories with WAV files found in {test_set_dir}\n"
            f"Expected structure: test_set/clean/*.wav, test_set/noise_10db/*.wav ..."
        )

    print(f"Found {len(condition_dirs)} condition folders:")
    for d in condition_dirs:
        wav_count = len(list(d.glob("*.wav")))
        print(f"  {d.name:25s}  {wav_count} WAV files")

    # Build task list: one entry per WAV file
    tasks = []
    for cond_dir in condition_dirs:
        out_cond_dir = output_dir / cond_dir.name
        out_cond_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in sorted(cond_dir.glob("*.wav")):
            npy_path = out_cond_dir / (wav_path.stem + ".npy")
            tasks.append((
                wav_path,
                npy_path,
                args.sample_rate,
                args.hop_length,
                args.n_fft,
                args.n_mels,
            ))

    total = len(tasks)
    print(f"\nTotal files to process: {total}")
    print(f"Workers: {args.num_workers}")
    print(f"Output:  {output_dir}\n")

    # Process in parallel
    errors = []
    skipped = 0
    done = 0

    with Pool(processes=args.num_workers) as pool:
        for path, status in tqdm(
            pool.imap_unordered(compute_mel_for_file, tasks),
            total=total,
            desc="Computing mels",
        ):
            if status == "skipped":
                skipped += 1
            elif status == "ok":
                done += 1
            else:
                errors.append((path, status))

    # Summary
    print(f"\nDone.")
    print(f"  Computed : {done}")
    print(f"  Skipped  : {skipped}  (already existed)")
    print(f"  Errors   : {len(errors)}")

    if errors:
        print("\nFailed files:")
        for path, err in errors:
            print(f"  {path}: {err}")

    # Verify output structure matches input
    print(f"\nOutput structure:")
    for cond_dir in condition_dirs:
        npy_count = len(list((output_dir / cond_dir.name).glob("*.npy")))
        wav_count = len(list(cond_dir.glob("*.wav")))
        status = "OK" if npy_count == wav_count else f"MISMATCH ({npy_count} npy vs {wav_count} wav)"
        print(f"  {cond_dir.name:25s}  {npy_count} .npy files  [{status}]")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute mel spectrograms from a fixed test set of WAV files."
    )
    parser.add_argument(
        "--test_set_dir", type=str, required=True,
        help="Root of the test set produced by build_test_set.py "
             "(contains clean/, noise_10db/, ... subdirectories)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write the .npy mel files (same subfolder structure)"
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--hop_length",  type=int, default=128)
    parser.add_argument("--n_fft",       type=int, default=2048)
    parser.add_argument("--n_mels",      type=int, default=512)
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count(),
        help="Number of parallel workers (default: all CPU cores)"
    )

    args = parser.parse_args()
    main(args)