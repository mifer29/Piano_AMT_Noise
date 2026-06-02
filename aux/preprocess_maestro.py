"""
Used to preprocess MIDI files from MAESTRO.
Adds the sustain pedal extension and fixes overlapping same-pitch notes, then extracts note events into a simple NumPy array format
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pretty_midi
from tqdm import tqdm
import sys
import os



# Sustain extension

def extend_notes_with_sustain(pm: pretty_midi.PrettyMIDI):
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        sustain_events = sorted(
            [cc for cc in instrument.control_changes if cc.number == 64],
            key=lambda x: x.time
        )

        if not sustain_events:
            continue

        sustain_on = False
        sustain_start = None
        intervals = []

        for cc in sustain_events:
            if cc.value >= 64 and not sustain_on:
                sustain_on = True
                sustain_start = cc.time
            elif cc.value < 64 and sustain_on:
                sustain_on = False
                intervals.append((sustain_start, cc.time))

        if sustain_on:
            intervals.append((sustain_start, pm.get_end_time()))

        for note in instrument.notes:
            for start, end in intervals:
                if start <= note.end <= end:
                    note.end = end
                    break

    return pm


# Fix overlapping same-pitch notes

def fix_note_overlaps(pm: pretty_midi.PrettyMIDI):
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        notes_by_pitch = {}

        for note in instrument.notes:
            notes_by_pitch.setdefault(note.pitch, []).append(note)

        for notes in notes_by_pitch.values():
            notes.sort(key=lambda n: n.start)

            for i in range(len(notes) - 1):
                if notes[i].end > notes[i + 1].start:
                    notes[i].end = notes[i + 1].start

    return pm


# Extract note array

def extract_note_array(pm: pretty_midi.PrettyMIDI):
    notes_list = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            notes_list.append([
                note.start,
                note.end,
                note.pitch,
                note.velocity
            ])

    if not notes_list:
        return np.zeros((0, 4), dtype=np.float32)

    notes_array = np.array(notes_list, dtype=np.float32)

    # Sort by onset time
    return notes_array[np.argsort(notes_array[:, 0])]



# Main processing

def process(maestro_root: str,
            output_dir: str,
            extend_sustain_flag: bool):

    maestro_root = Path(maestro_root).resolve()
    output_dir = Path(output_dir).resolve()

    print(f"MAESTRO ROOT: {maestro_root}", flush=True)
    print(f"OUTPUT DIR: {output_dir}", flush=True)
    print(f"Extend sustain: {extend_sustain_flag}", flush=True)

    if not maestro_root.exists():
        print(" MAESTRO root directory does not exist.", flush=True)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = maestro_root / "maestro-v3.0.0.csv"

    if not metadata_csv.exists():
        print(" maestro-v3.0.0.csv not found.", flush=True)
        sys.exit(1)

    metadata = pd.read_csv(metadata_csv)

    print(f"\nFound {len(metadata)} MIDI files.", flush=True)
    print("Starting preprocessing...\n", flush=True)

    processed = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(metadata.iterrows(),
                        total=len(metadata),
                        desc="Processing MIDIs"):

        midi_path = maestro_root / row["midi_filename"]

        if not midi_path.exists():
            continue

        output_path = output_dir / (midi_path.stem + "_events.npy")

        # Resume-safe skip
        if output_path.exists():
            skipped += 1
            continue

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))

            if extend_sustain_flag:
                pm = extend_notes_with_sustain(pm)
                pm = fix_note_overlaps(pm)

            notes_array = extract_note_array(pm)

            np.save(output_path, notes_array)

            processed += 1

        except Exception as e:
            print(f"Error processing {midi_path.name}: {e}", flush=True)
            failed += 1

    print("\nPreprocessing finished.", flush=True)
    print(f"Processed: {processed}", flush=True)
    print(f"Skipped:   {skipped}", flush=True)
    print(f"Failed:    {failed}", flush=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--maestro_root",
        type=str,
        required=True,
        help="Path to MAESTRO root directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where .npy event files will be saved"
    )

    parser.add_argument(
        "--extend_sustain",
        action="store_true",
        help="Apply sustain pedal extension"
    )

    args = parser.parse_args()

    process(
        maestro_root=args.maestro_root,
        output_dir=args.output_dir,
        extend_sustain_flag=args.extend_sustain
    )