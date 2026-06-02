'''
Builds a single HDF5 test file from the pre-generated mel spectrograms produced by precompute_test_mels.py

The output HDF5 contains one mel dataset per condition, all sharing the same event annotations and length arrays:

  mel_clean              (total_frames, n_mels)  float16
  mel_noise_20db         (total_frames, n_mels)  float16
  mel_noise_10db         (total_frames, n_mels)  float16
  mel_noise_5db          (total_frames, n_mels)  float16
  mel_speech_15db        (total_frames, n_mels)  float16
  mel_reverb             (total_frames, n_mels)  float16
  mel_phone_eq           (total_frames, n_mels)  float16
  mel_phone_simulation   (total_frames, n_mels)  float16
  events                 (total_events, 4)        float32
  mel_lengths            (n_pieces,)              int64
  event_lengths          (n_pieces,)              int64

Song order is fixed by manifest.csv (produced by build_test_set.py), which guarantees reproducibility across machines
'''


import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


# Conditions must match the subfolder names written by precompute_test_mels.py
CONDITIONS = [
    "clean",
    "noise_20db",
    "noise_10db",
    "noise_5db",
    "speech_15db",
    "reverb_large",
    "reverb_small",
    "phone_eq",
    "phone_simulation",
]


def build_test_hdf5(
    maestro_root: str,
    events_dir: str,
    test_mels_dir: str,
    manifest_path: str,
    output_path: str,
):
    maestro_root  = Path(maestro_root)
    events_dir    = Path(events_dir)
    test_mels_dir = Path(test_mels_dir)
    manifest_path = Path(manifest_path)

    # Load manifest: this defines song order, which must be stable
    # manifest.csv was written by build_test_set.py and contains exactly
    # the songs that were degraded.  Using it as the source of truth means
    # song order is identical to the WAV and mel files on disk.

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(manifest)} songs")

    # Verify all condition folders exist
    for cond in CONDITIONS:
        cond_dir = test_mels_dir / cond
        if not cond_dir.exists():
            raise FileNotFoundError(
                f"Condition folder not found: {cond_dir}\n"
                f"Run precompute_test_mels.py first."
            )

    # Pass 1: scan sizes

    print("\nPass 1: scanning file sizes ...")

    mel_lengths   = []
    event_lengths = []
    mel_dim       = None

    for i, row in manifest.iterrows():
        midi_stem = Path(row["midi_filename"]).stem
        audio_stem = Path(row["audio_filename"]).stem

        # All conditions share the same mel shape — use clean as reference
        mel_path = test_mels_dir / "clean" / f"{audio_stem}.npy"
        if not mel_path.exists():
            raise FileNotFoundError(
                f"Clean mel not found: {mel_path}\n"
                f"Check that precompute_test_mels.py completed successfully."
            )

        mel = np.load(str(mel_path), mmap_mode="r")

        events_path = events_dir / f"{audio_stem}.npy"
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")

        events = np.load(str(events_path), mmap_mode="r")

        mel_lengths.append(mel.shape[0])
        event_lengths.append(events.shape[0])

        if mel_dim is None:
            mel_dim = mel.shape[1]
        elif mel.shape[1] != mel_dim:
            raise ValueError(
                f"Mel dimension mismatch at {mel_path}: "
                f"expected {mel_dim}, got {mel.shape[1]}"
            )

        if (i + 1) % 5 == 0 or (i + 1) == len(manifest):
            print(f"  Scanned {i + 1}/{len(manifest)} pieces")

    total_mel_frames = sum(mel_lengths)
    total_event_rows = sum(event_lengths)

    print(f"\n  Songs        : {len(manifest)}")
    print(f"  Conditions   : {len(CONDITIONS)}")
    print(f"  Mel frames   : {total_mel_frames}  (per condition)")
    print(f"  Event rows   : {total_event_rows}")
    print(f"  Mel dim      : {mel_dim}")
    print(f"  Est. HDF5 size: "
          f"~{total_mel_frames * mel_dim * 2 * len(CONDITIONS) / 1e9:.2f} GB "
          f"(float16, all conditions)")

    # Pass 2: write HDF5

    print(f"\nPass 2: writing {output_path} ...")

    with h5py.File(output_path, "w", libver="latest") as f:

        # Create one mel dataset per condition
        mel_datasets = {}
        for cond in CONDITIONS:
            ds_name = f"mel_{cond}"
            mel_datasets[cond] = f.create_dataset(
                ds_name,
                shape=(total_mel_frames, mel_dim),
                dtype="float16",
                chunks=(4096, mel_dim),
            )
            print(f"  Created dataset: {ds_name}  shape=({total_mel_frames}, {mel_dim})")

        # Shared event dataset — annotations are condition-independent
        event_ds = f.create_dataset(
            "events",
            shape=(total_event_rows, 4),
            dtype="float32",
            chunks=(4096, 4),
        )

        # Length arrays — shared across all conditions
        f.create_dataset("mel_lengths",   data=np.array(mel_lengths,   dtype=np.int64))
        f.create_dataset("event_lengths", data=np.array(event_lengths, dtype=np.int64))

        # Store condition list as metadata so evaluation code can read it
        f.attrs["conditions"]  = CONDITIONS
        f.attrs["num_songs"]   = len(manifest)
        f.attrs["mel_dim"]     = mel_dim

        print(f"\nWriting data ...")

        mel_cursor   = 0
        event_cursor = 0

        for i, row in manifest.iterrows():
            midi_stem  = Path(row["midi_filename"]).stem
            audio_stem = Path(row["audio_filename"]).stem

            mel_len   = mel_lengths[i]
            event_len = event_lengths[i]

            # Write mels for every condition at this song's slice
            for cond in CONDITIONS:
                mel_path = test_mels_dir / cond / f"{audio_stem}.npy"
                mel = np.load(str(mel_path))

                if mel.dtype != np.float16:
                    mel = mel.astype(np.float16)

                mel_datasets[cond][mel_cursor : mel_cursor + mel_len] = mel

            # Write events (same for all conditions)
            events_path = events_dir / f"{midi_stem}.npy"
            events = np.load(str(events_path)).astype(np.float32)
            event_ds[event_cursor : event_cursor + event_len] = events

            mel_cursor   += mel_len
            event_cursor += event_len

            if (i + 1) % 5 == 0 or (i + 1) == len(manifest):
                print(f"  Wrote {i + 1}/{len(manifest)} pieces  "
                      f"(mel cursor: {mel_cursor}, event cursor: {event_cursor})")

    print(f"\nFinished: {output_path}")
    print(f"  Datasets written: {[f'mel_{c}' for c in CONDITIONS] + ['events', 'mel_lengths', 'event_lengths']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build test HDF5 with one mel dataset per noise condition."
    )
    parser.add_argument(
        "--maestro_root", type=str, required=True,
        help="MAESTRO root directory (contains maestro-v3.0.0.csv)"
    )
    parser.add_argument(
        "--events_dir", type=str, required=True,
        help="Directory containing {stem}_events.npy files"
    )
    parser.add_argument(
        "--test_mels_dir", type=str, required=True,
        help="Root of precomputed test mels (contains clean/, noise_10db/, ... subfolders)"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.csv produced by build_test_set.py"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path for the output HDF5 file"
    )

    args = parser.parse_args()

    build_test_hdf5(
        maestro_root  = args.maestro_root,
        events_dir    = args.events_dir,
        test_mels_dir = args.test_mels_dir,
        manifest_path      = args.manifest,
        output_path   = args.output_path,
    )