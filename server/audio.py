import logging
import numpy as np
import librosa

from config import SAMPLE_RATE, HOP_LENGTH, N_MELS
from vocab import token_to_id

log = logging.getLogger("transcribe")



# Audio to mel
def compute_mel(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_fft=2048, n_mels=N_MELS
    )
    return np.log(mel + 1e-6).T.astype(np.float32)

def get_audio_duration(audio_path):
    """Return audio duration in seconds, or -1 if it can't be probed."""
    try:
        return librosa.get_duration(path=audio_path)
    except Exception as e:
        log.warning(f"Could not probe audio duration: {e}")
        return -1.0



# Tokens to notes

def tokens_to_notes(token_ids, time_resolution=0.01, segment_duration=None):
    """
    Decode a token sequence into (onset, offset, pitch, velocity) tuples.

    Token order (matches training, see maestro_dataset._events_to_tokens):
        attack:  time -> note_on -> velocity
        release: time -> note_off

    Because velocity tokens always follow the note_on they apply to, this
    function peeks ahead one position after each note_on rather than
    maintaining a running 'current velocity' state. The previous
    running-state implementation produced off-by-one velocities (each note
    received the previous note's velocity).
    """
    id_to_tok    = {i: t for t, i in token_to_id.items()}
    current_time = 0.0
    active_notes = {}   # pitch -> (onset_time, velocity)
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
            try:
                current_time = int(tok.split("_")[1]) * time_resolution
            except (ValueError, IndexError):
                log.warning(f"Malformed time token '{tok}' at position {i}, skipping")
            i += 1

        elif tok.startswith("note_on_"):
            try:
                pitch = int(tok.split("_")[2])
            except (ValueError, IndexError):
                log.warning(f"Malformed note_on token '{tok}' at position {i}, skipping")
                i += 1
                continue

            # Close any overlapping note of the same pitch
            if pitch in active_notes:
                onset, vel = active_notes.pop(pitch)
                notes.append((onset, current_time, pitch, vel))

            # Peek ahead for the velocity token that should follow
            vel_bin = 0
            if i + 1 < len(token_ids):
                next_tok = id_to_tok.get(int(token_ids[i + 1]), "")
                if next_tok.startswith("velocity_"):
                    try:
                        vel_bin = int(next_tok.split("_")[1])
                        i += 1  # consume the velocity token
                    except (ValueError, IndexError):
                        log.warning(f"Malformed velocity token '{next_tok}', using 0")

            midi_vel = max(1, vel_bin)
            active_notes[pitch] = (current_time, midi_vel)
            i += 1

        elif tok.startswith("note_off_"):
            try:
                pitch = int(tok.split("_")[2])
            except (ValueError, IndexError):
                log.warning(f"Malformed note_off token '{tok}' at position {i}, skipping")
                i += 1
                continue

            if pitch in active_notes:
                onset, vel = active_notes.pop(pitch)
                notes.append((onset, current_time, pitch, vel))
            i += 1

        elif tok.startswith("velocity_"):
            # Orphaned velocity (not preceded by a note_on) — skip
            i += 1

        else:
            i += 1

    # Close any notes that remain open at the segment boundary
    close_time = segment_duration if segment_duration is not None else current_time
    for pitch, (onset, vel) in active_notes.items():
        if close_time > onset:
            notes.append((onset, close_time, pitch, vel))

    return notes


# Deduplication of overlap artifacts

def deduplicate_notes(notes, time_tol=0.05):
    """
    Merge duplicated notes caused by overlapping segments.
    Keeps the longest note when duplicates are found.

    Identical to evaluate.py:deduplicate_notes so the deployed pipeline
    resolves boundary duplicates exactly as the evaluation pipeline did.
    """
    if len(notes) == 0:
        return notes

    notes  = sorted(notes, key=lambda x: (x[2], x[0]))  # pitch, onset
    merged = []

    for note in notes:
        onset, offset, pitch, velocity = note

        if not merged:
            merged.append(note)
            continue

        p_on, p_off, p_pitch, p_vel = merged[-1]

        if pitch == p_pitch and abs(onset - p_on) < time_tol:
            # keep longer note
            if offset > p_off:
                merged[-1] = (p_on, offset, pitch, velocity)
        else:
            merged.append(note)

    return merged
