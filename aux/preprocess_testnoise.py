"""
Preprocesses test MIDI files with the same sustain extension logic used to build the MAESTRO HDF5, so ground truth is consistent with training.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pretty_midi
from tqdm import tqdm



# Sustain extension


def extend_notes_with_sustain(pm):
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



# Fix overlaps


def fix_note_overlaps(pm):
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



# Extract notes


def extract_note_array(pm):
    notes = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            if note.end <= note.start:
                continue

            notes.append([
                note.start,
                note.end,
                note.pitch,
                note.velocity
            ])

    if not notes:
        return np.zeros((0, 4), dtype=np.float32)

    notes = np.array(notes, dtype=np.float32)
    return notes[np.argsort(notes[:, 0])]




def main(args):

    maestro_root = Path(args.maestro_root)
    test_set_dir = Path(args.test_set_dir)
    output_dir   = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest 
    manifest = pd.read_csv(test_set_dir / "manifest.csv")

    print(f"Processing {len(manifest)} test songs")

    processed = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):

        midi_path = maestro_root / row["midi_filename"]
        stem      = Path(row["audio_filename"]).stem

        out_path = output_dir / f"{stem}.npy"

        if out_path.exists():
            continue

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))

            
            pm = extend_notes_with_sustain(pm)
            pm = fix_note_overlaps(pm)

            notes = extract_note_array(pm)

            np.save(out_path, notes)

            processed += 1

        except Exception as e:
            print(f"Error: {midi_path} → {e}")

    print(f"\nDone. Processed: {processed}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--maestro_root", type=str, required=True)
    parser.add_argument("--test_set_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)