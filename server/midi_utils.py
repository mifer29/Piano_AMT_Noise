import logging
import numpy as np
import pretty_midi

log = logging.getLogger("transcribe")



# Notes to MIDI

def notes_to_midi(notes, path):
    pm   = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for o, f, p, v in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=int(np.clip(v, 1, 127)),
            pitch=int(p),
            start=float(o),
            end=float(max(f, o + 0.01))
        ))
    pm.instruments.append(inst)
    pm.write(path)
    return pm

def safe_estimate_tempo(pm, default_bpm=120):
    """estimate_tempo() can throw or return NaN/inf on sparse MIDI.
    Fall back to a sensible default rather than crashing the request."""
    try:
        bpm = float(pm.estimate_tempo())
        if not np.isfinite(bpm) or bpm <= 0 or bpm > 400:
            log.warning(f"estimate_tempo() returned invalid value {bpm}, using default {default_bpm}")
            return float(default_bpm)
        return bpm
    except Exception as e:
        log.warning(f"estimate_tempo() failed: {e}, using default {default_bpm}")
        return float(default_bpm)
