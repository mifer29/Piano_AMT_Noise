"""
Generate MIDI + WAV from model output and compare with original

For each selected song produces:
  song_NNN/
    predicted.mid: model output as MIDI
    predicted.wav: model output synthesized to audio
    original.wav: copy of the original recording

Requirements:
    pip install pretty_midi scipy --break-system-packages
    # Also needs a .sf2 soundfont for WAV synthesis.
    # Free option: apt-get install fluid-soundfont-gm
    # Then use: /usr/share/sounds/sf2/FluidR3_GM.sf2

"""

import os
import sys
import csv
import time
import shutil
import argparse
import traceback

import numpy as np
import torch
import scipy.io.wavfile as wavfile
import pretty_midi

from maestro_dataset import MAESTROSeq2SeqDataset
from model import AudioTransformer, greedy_decode



# Token to note conversion

def tokens_to_notes(token_ids, token_to_id, time_resolution=0.01):
    id_to_tok    = {i: t for t, i in token_to_id.items()}
    current_time = 0.0
    current_vel  = 0
    active_notes = {}
    notes        = []

    i = 0
    while i < len(token_ids):
        tok = id_to_tok.get(int(token_ids[i]), "UNKNOWN")

        if tok == "EOS":
            break
        if tok in ("PAD", "BOS"):
            i += 1
            continue

        if tok.startswith("time_"):
            current_time = int(tok.split("_")[1]) * time_resolution
            i += 1
        elif tok.startswith("velocity_"):
            current_vel = int(tok.split("_")[1])
            i += 1
        elif tok.startswith("note_"):
            pitch = int(tok.split("_")[1])
            if current_vel == 0:
                if pitch in active_notes:
                    onset, vel = active_notes.pop(pitch)
                    notes.append((onset, current_time, pitch, vel))
            else:
                if pitch in active_notes:
                    onset, vel = active_notes.pop(pitch)
                    notes.append((onset, current_time, pitch, vel))
                active_notes[pitch] = (current_time, current_vel)
            i += 1
        else:
            i += 1

    # Force-close any notes still active
    for pitch, (onset, vel) in active_notes.items():
        notes.append((onset, current_time + 0.05, pitch, vel))

    return notes


# --------------------------------------------------
# Per-song transcription
# --------------------------------------------------

@torch.no_grad()
def transcribe_song(model, dataset, song_idx, device, max_output_tokens=512,
                    max_segments=20, log=None):

    def _log(msg):
        if log: log(f"  [transcribe] {msg}")

    f          = dataset._get_file()
    mel_start  = int(dataset.mel_offsets[song_idx])
    mel_len    = int(dataset.mel_lengths[song_idx])
    max_frames = dataset.max_input_frames
    hop        = dataset.hop_length
    sr         = dataset.sample_rate
    seg_dur    = max_frames * hop / sr

    total_segs = int(np.ceil(mel_len / max_frames))
    run_segs   = min(total_segs, max_segments)
    _log(f"mel_len={mel_len}, {total_segs} total segments of {seg_dur:.2f}s — running first {run_segs}")

    all_notes   = []
    time_offset = 0.0
    frame_pos   = 0
    seg_idx     = 0

    while frame_pos < mel_len and seg_idx < max_segments:

        mel = f["mel"][mel_start + frame_pos : mel_start + frame_pos + max_frames]

        if mel.shape[0] < max_frames:
            pad = np.zeros((max_frames - mel.shape[0], dataset.n_mels), dtype=np.float32)
            mel = np.vstack([mel, pad])

        if mel.dtype != np.float32:
            mel = mel.astype(np.float32)

        audio_t   = torch.from_numpy(mel).unsqueeze(0).to(device)
        token_ids = greedy_decode(
            model, audio_t,
            bos_id=dataset.bos_id,
            eos_id=dataset.eos_id,
            max_len=max_output_tokens
        )

        seg_notes = tokens_to_notes(token_ids, dataset.token_to_id)
        for onset, offset, pitch, velocity in seg_notes:
            if onset <= seg_dur:
                all_notes.append((
                    onset  + time_offset,
                    offset + time_offset,
                    pitch,
                    velocity,
                ))

        _log(f"Segment {seg_idx}: {len(seg_notes)} notes")
        frame_pos   += max_frames
        time_offset += seg_dur
        seg_idx     += 1

    _log(f"Total: {len(all_notes)} notes across {seg_idx} segments")
    return all_notes


# --------------------------------------------------
# Notes → PrettyMIDI object
# --------------------------------------------------

def notes_to_pretty_midi(notes):
    pm    = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    for onset, offset, pitch, velocity in sorted(notes, key=lambda x: x[0]):
        pitch    = int(np.clip(pitch,    0, 127))
        velocity = int(np.clip(velocity, 1, 127))
        offset   = max(offset, onset + 0.01)

        piano.notes.append(pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=float(onset),
            end=float(offset),
        ))

    pm.instruments.append(piano)
    return pm


# --------------------------------------------------
# CSV → ordered validation wav paths
# --------------------------------------------------

def load_validation_wav_paths(csv_path, wav_root):
    """
    Returns wav paths for the validation split in CSV order,
    which matches the HDF5 song index order.
    """
    paths = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "validation":
                paths.append(os.path.join(wav_root, row["audio_filename"]))
    return paths


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "compare.log")

    def log(msg):
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("AMT COMPARISON")
    log(f"Checkpoint:   {args.checkpoint}")
    log(f"HDF5 val:     {args.hdf5_val}")
    log(f"CSV:          {args.maestro_csv}")
    log(f"WAV root:     {args.wav_root}")
    log(f"Soundfont:    {args.soundfont or 'NOT PROVIDED — predicted.wav will be skipped'}")
    log(f"Song indices: {args.song_indices}")
    log(f"Max segments: {args.max_segments} (~{args.max_segments * 4:.0f}s per song)")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # --------------------------------------------------
    # WAV index from CSV
    # --------------------------------------------------

    log("Building WAV index from CSV...")
    wav_paths = load_validation_wav_paths(args.maestro_csv, args.wav_root)
    log(f"Found {len(wav_paths)} validation songs in CSV")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    log("Loading validation dataset...")
    dataset = MAESTROSeq2SeqDataset(
        hdf5_path=args.hdf5_val,
        split="validation",
        max_output_tokens=512,
    )
    log(f"Dataset: {len(dataset)} songs, vocab: {len(dataset.vocab)} tokens")

    if len(wav_paths) != len(dataset):
        log(f"WARNING: CSV has {len(wav_paths)} songs but HDF5 has {len(dataset)} — index may be misaligned")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    log("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    step       = checkpoint.get("step", "unknown")
    log(f"Step: {step}")

    model = AudioTransformer(n_mels=512, vocab_size=len(dataset.vocab))
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    log(f"Model ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # --------------------------------------------------
    # Per-song loop
    # --------------------------------------------------

    for song_idx in args.song_indices:

        log(f"{'='*50}")
        log(f"Song {song_idx}")

        if song_idx >= len(dataset):
            log(f"  SKIP: index out of range")
            continue

        song_dir = os.path.join(args.output_dir, f"song_{song_idx:03d}")
        os.makedirs(song_dir, exist_ok=True)

        # 1. Copy original WAV
        if song_idx < len(wav_paths):
            wav_src = wav_paths[song_idx]
            log(f"  Original: {wav_src}")
            if os.path.exists(wav_src):
                shutil.copy2(wav_src, os.path.join(song_dir, "original.wav"))
                log(f"  Copied original.wav")
            else:
                log(f"  WARNING: file not found: {wav_src}")
        else:
            log(f"  WARNING: no WAV path for song {song_idx}")

        # 2. Transcribe
        log(f"  Transcribing...")
        t0 = time.time()
        try:
            notes = transcribe_song(model, dataset, song_idx, device,
                                    max_segments=args.max_segments, log=log)
            log(f"  Done in {time.time()-t0:.1f}s — {len(notes)} notes")
        except Exception as e:
            log(f"  FAILED: {e}")
            log(traceback.format_exc())
            continue

        if len(notes) == 0:
            log("  WARNING: 0 notes predicted")

        # 3. Build PrettyMIDI object
        pm = notes_to_pretty_midi(notes)

        # 4. Save MIDI
        midi_path = os.path.join(song_dir, "predicted.mid")
        pm.write(midi_path)
        log(f"  Saved predicted.mid")

        # 5. Synthesize to WAV using pretty_midi's built-in fluidsynth
        if args.soundfont:
            log(f"  Synthesizing predicted.wav...")
            try:
                audio = pm.fluidsynth(fs=16000, sf2_path=args.soundfont)

                # Normalize and convert to int16
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                audio_int16 = np.int16(audio * 32767)

                wav_path = os.path.join(song_dir, "predicted.wav")
                wavfile.write(wav_path, 16000, audio_int16)
                log(f"  Saved predicted.wav ({len(audio)/16000:.1f}s)")
            except Exception as e:
                log(f"  WAV synthesis failed: {e}")
                log(traceback.format_exc())
                log(f"  It can still be converted manually:")
                log(f"    fluidsynth -ni -F {song_dir}/predicted.wav {args.soundfont} {midi_path}")
        else:
            log("  Skipping WAV synthesis (no --soundfont)")
            log(f"  To synthesize manually:")
            log(f"    fluidsynth -ni -F {song_dir}/predicted.wav /path/to/soundfont.sf2 {midi_path}")

        # Summary
        log(f"  Files in {song_dir}:")
        for fname in sorted(os.listdir(song_dir)):
            size = os.path.getsize(os.path.join(song_dir, fname)) / 1e6
            log(f"    {fname}  ({size:.2f} MB)")

    log("=" * 60)
    log("DONE")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--hdf5_val",     type=str, required=True)
    parser.add_argument("--maestro_csv",  type=str, required=True)
    parser.add_argument("--wav_root",     type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--song_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max_segments", type=int, default=20,
                        help="Max 4s segments to transcribe per song (default: 20 = ~80s)")
    parser.add_argument("--soundfont",    type=str, default=None,
                        help="Path to .sf2 soundfont for WAV synthesis. "
                             "Free option: /usr/share/sounds/sf2/FluidR3_GM.sf2")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)