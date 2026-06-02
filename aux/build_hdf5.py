'''
Four datasets are created inside the HDF5 file: 
    - mel: mel frames concatenated
    - events: note events concatenadted
    - mel_lenghts: length of mels per piece
    - event_lengths: length of events per piece

'''

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

def build_split(maestro_root, events_dir, mel_dir, split, output_path):

    maestro_root = Path(maestro_root)
    events_dir = Path(events_dir)
    mel_dir = Path(mel_dir)

    metadata = pd.read_csv(maestro_root / "maestro-v3.0.0.csv")
    split_df = metadata[metadata["split"] == split]

    print(f"\nBuilding HDF5 for split: {split}")
    print(f"Total pieces: {len(split_df)}")

    mel_lengths = []
    event_lengths = []

    # Computing total sizes

    total_mel_frames = 0
    total_event_rows = 0

    for i, row in split_df.iterrows():

        midi_stem = Path(row["midi_filename"]).stem
        mel_stem = Path(row["audio_filename"]).stem

        mel = np.load(mel_dir / f"{mel_stem}_mel.npy", mmap_mode="r")
        events = np.load(events_dir / f"{midi_stem}_events.npy", mmap_mode="r")

        mel_lengths.append(mel.shape[0])
        event_lengths.append(events.shape[0])

        total_mel_frames += mel.shape[0]
        total_event_rows += events.shape[0]

        if i % 100 == 0:
            print(f"Scanned {i}/{len(split_df)} pieces...")

    mel_dim = mel.shape[1]

    print(f"Total mel frames: {total_mel_frames}")
    print(f"Total event rows: {total_event_rows}")


    # Creating HDF5 datasets


    with h5py.File(output_path, "w", libver="latest") as f:

        # Use float16 to avoid doubling disk size
        mel_ds = f.create_dataset(
            "mel",
            shape=(total_mel_frames, mel_dim),
            dtype="float16",
            chunks=(4096, mel_dim),  # larger chunks = faster build
        )

        event_ds = f.create_dataset(
            "events",
            shape=(total_event_rows, 4),
            dtype="float32",
            chunks=(4096, 4),
        )

        f.create_dataset("mel_lengths", data=np.array(mel_lengths))
        f.create_dataset("event_lengths", data=np.array(event_lengths))

        # Write incrementally now

        mel_cursor = 0
        event_cursor = 0

        for i, row in split_df.iterrows():

            midi_stem = Path(row["midi_filename"]).stem
            mel_stem = Path(row["audio_filename"]).stem

            mel = np.load(mel_dir / f"{mel_stem}_mel.npy")
            events = np.load(events_dir / f"{midi_stem}_events.npy")

            mel_len = mel.shape[0]
            event_len = events.shape[0]

            # Convert mel to float16 BEFORE writing
            if mel.dtype != np.float16:
                mel = mel.astype(np.float16)

            mel_ds[mel_cursor:mel_cursor + mel_len] = mel
            event_ds[event_cursor:event_cursor + event_len] = events.astype(np.float32)

            mel_cursor += mel_len
            event_cursor += event_len

            if i % 25 == 0:
                print(f"Wrote {i}/{len(split_df)} pieces...")

    print(f"\nFinished building {output_path}")


if __name__ == "__main__":

    build_split(
            maestro_root=os.getenv("maestro_root"),
            events_dir=os.getenv("events_dir"),
            mel_dir=os.getenv("mel_dir"),
            split="test",
            output_path=os.getenv("output_path")
        )

'''
    build_split(
        maestro_root=os.getenv("maestro_root"),
        events_dir=os.getenv("events_dir"),
        mel_dir=os.getenv("mel_dir"),
        split="train",
        output_path=os.getenv("output_path")
    )

    build_split(
        maestro_root=os.getenv("maestro_root"),
        events_dir=os.getenv("events_dir"),
        mel_dir=os.getenv("mel_dir"),
        split="validation",
        output_path=os.getenv("output_path")
    )
    '''