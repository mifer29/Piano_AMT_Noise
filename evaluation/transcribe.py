"""
Transcribes a WAV file to MIDI using the trained AMT models.
Uses the same mel pipeline as the MAESTRO preprocessing script.
"""

import argparse
import time
import numpy as np
import torch
import librosa
import soundfile as sf
import pretty_midi

from model import AudioTransformer, greedy_decode


# Manual vocabulary construction (must match training)

time_tokens = 1000 # This must be changed if the vocab is modified to a different time resolution

def build_vocab(max_time_bins=time_tokens, velocity_bins=128):

    special_tokens = ["PAD", "BOS", "EOS"]

    time_tokens = [f"time_{i}" for i in range(max_time_bins)]
    velocity_tokens = [f"velocity_{i}" for i in range(velocity_bins)]

    note_tokens = (
        [f"note_on_{i}" for i in range(128)] +
        [f"note_off_{i}" for i in range(128)]
    )

    vocab = special_tokens + time_tokens + velocity_tokens + note_tokens
    token_to_id = {tok: i for i, tok in enumerate(vocab)}

    return vocab, token_to_id



# Mel spectrogram


def compute_mel(audio_path,
                sample_rate=16000,
                hop_length=128,
                n_fft=2048,
                n_mels=512):

    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        power=2.0,
    )

    mel = np.log(mel + 1e-6)
    mel = mel.T.astype(np.float32)

    print(f"Mel shape: {mel.shape}")
    return mel



# Tokens to notes

def tokens_to_notes(token_ids, token_to_id, time_resolution=0.01):

    id_to_tok = {i: t for t, i in token_to_id.items()}

    current_time = 0.0
    active_notes = {}
    notes = []

    i = 0
    while i < len(token_ids):

        tok = id_to_tok.get(int(token_ids[i]), "UNK")

        if tok == "EOS":
            break

        if tok in ("PAD", "BOS"):
            i += 1
            continue

        if tok.startswith("time_"):
            current_time = int(tok.split("_")[1]) * time_resolution
            i += 1

        elif tok.startswith("note_on_"):
            pitch = int(tok.split("_")[2])

            # Close overlapping
            if pitch in active_notes:
                onset, vel = active_notes.pop(pitch)
                notes.append((onset, current_time, pitch, vel))

            vel_bin = 0
            if i + 1 < len(token_ids):
                next_tok = id_to_tok.get(int(token_ids[i + 1]), "")
                if next_tok.startswith("velocity_"):
                    vel_bin = int(next_tok.split("_")[1])
                    i += 1

            

            midi_vel = max(1, vel_bin)

            active_notes[pitch] = (current_time, midi_vel)
            i += 1

        elif tok.startswith("note_off_"):
            pitch = int(tok.split("_")[2])

            if pitch in active_notes:
                onset, vel = active_notes.pop(pitch)
                notes.append((onset, current_time, pitch, vel))

            i += 1

        else:
            i += 1

    # Close remaining
    for pitch, (onset, vel) in active_notes.items():
        notes.append((onset, current_time + 0.05, pitch, vel))

    return notes



# Sliding inference


@torch.no_grad()
def transcribe_mel(model,
                   mel,
                   token_to_id,
                   bos_id,
                   eos_id,
                   max_input_frames=512,
                   hop_length=128,
                   sample_rate=16000,
                   max_output_tokens=1024,
                   device="cpu"):

    stride      = int(max_input_frames * 0.5)
    stride_time = stride * hop_length / sample_rate

    mel_len     = mel.shape[0]
    seg_dur     = max_input_frames * hop_length / sample_rate

    all_notes   = []
    frame_pos   = 0
    time_offset = 0.0

    while frame_pos < mel_len:

        seg = mel[frame_pos : frame_pos + max_input_frames]

        if seg.shape[0] < max_input_frames:
            pad = np.zeros((max_input_frames - seg.shape[0], mel.shape[1]))
            seg = np.vstack([seg, pad])

        audio_t   = torch.from_numpy(seg).unsqueeze(0).float().to(device)
        token_ids = greedy_decode(
            model, audio_t,
            bos_id=bos_id, eos_id=eos_id,
            max_len=max_output_tokens,
        )

        seg_notes = tokens_to_notes(token_ids, token_to_id)

        for onset, offset, pitch, velocity in seg_notes:
            if onset <= seg_dur:
                all_notes.append((
                    onset  + time_offset,
                    offset + time_offset,
                    pitch,
                    velocity,
                ))

        frame_pos   += stride
        time_offset += stride_time

    all_notes = sorted(all_notes, key=lambda x: x[0])
    all_notes = deduplicate_notes(all_notes)       # ← add this

    print(f"Total notes: {len(all_notes)}")
    return all_notes


def deduplicate_notes(notes, time_tol=0.05):
    """
    Merge notes of the same pitch whose onsets are within time_tol seconds.
    Keeps the longest version: same logic as evaluate.py.
    """
    if not notes:
        return notes

    notes  = sorted(notes, key=lambda x: (x[2], x[0]))   # pitch, onset
    merged = []

    for note in notes:
        onset, offset, pitch, velocity = note

        if not merged:
            merged.append(note)
            continue

        p_on, p_off, p_pitch, p_vel = merged[-1]

        if pitch == p_pitch and abs(onset - p_on) < time_tol:
            if offset > p_off:                  # keep the longer one
                merged[-1] = (p_on, offset, pitch, velocity)
        else:
            merged.append(note)

    return merged



# MIDI writer


def notes_to_midi(notes, output_path):

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for onset, offset, pitch, velocity in notes:
        note = pretty_midi.Note(
            velocity=int(np.clip(velocity, 1, 127)),
            pitch=int(pitch),
            start=float(onset),
            end=float(max(offset, onset + 0.01)),
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_path)

    print(f"Saved MIDI: {output_path}")



# Main


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab, token_to_id = build_vocab()

    bos_id = token_to_id["BOS"]
    eos_id = token_to_id["EOS"]

    print(f"Vocab size: {len(vocab)}")

    mel = compute_mel(args.input)

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # detect embedding key automatically
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.", ""): v.float() for k, v in state_dict.items()}

    embedding_key = [k for k in state_dict if "embedding.weight" in k][0]
    ckpt_vocab_size = state_dict[embedding_key].shape[0]

    print(f"Checkpoint vocab size: {ckpt_vocab_size}")

    assert ckpt_vocab_size == len(vocab), \
        f" VOCAB MISMATCH: checkpoint={ckpt_vocab_size}, current={len(vocab)}"

    model = AudioTransformer(
        n_mels=512,
        vocab_size=len(vocab)
    )

    model.load_state_dict(state_dict)
    model = model.to(device).float()
    model.eval()

    t0 = time.time()

    notes = transcribe_mel(
        model,
        mel,
        token_to_id,
        bos_id,
        eos_id,
        device=device
    )

    print(f"Inference time: {time.time()-t0:.2f}s")

    notes_to_midi(notes, args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(args)