"""
Piano AMT model evaluation using mir_eval and mpteval

Evaluating on MAESTRO test set, with GT.
"""

import os
import sys
import json
import time
import argparse
import traceback

import numpy as np
import torch
import mir_eval
import pretty_midi

import mpteval

from maestro_dataset import MAESTROSeq2SeqDataset
from model import AudioTransformer, greedy_decode



# mir_eval version detection: just in case the API changes
# this is important becuase there are some versions with different names for the velocity functionalities

def check_mir_eval_version(log):
    log(f"mir_eval version: {mir_eval.__version__}")

    # Check which velocity API is available
    has_velocity_tolerance = "velocity_tolerance" in str(
        mir_eval.transcription.precision_recall_f1_overlap.__doc__ or ""
    )
    has_transcription_velocity = hasattr(mir_eval, "transcription_velocity")
    log(f"  has velocity_tolerance kwarg: {has_velocity_tolerance}")
    log(f"  has transcription_velocity module: {has_transcription_velocity}")
    return has_velocity_tolerance, has_transcription_velocity



# Token to note event conversion

def tokens_to_notes(token_ids, token_to_id, time_resolution=0.01, segment_duration=None, log=None):
    # Note: We accept segment_duration so open notes at segment end are closed
    # at the true boundary, not at whatever the last time token happened to be.

    def _log(msg):
        if log:
            log(f"    [tokens_to_notes] {msg}")

    id_to_tok    = {i: t for t, i in token_to_id.items()}
    current_time = 0.0
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

        elif tok.startswith("note_on_"):
            pitch = int(tok.split("_")[2])

            # close overlapping note
            if pitch in active_notes:
                onset, vel = active_notes.pop(pitch)
                notes.append((onset, current_time, pitch, vel))

            # peek ahead for velocity (new order: note_on → velocity)
            vel_bin = 0
            if i + 1 < len(token_ids):
                next_tok = id_to_tok.get(int(token_ids[i + 1]), "")
                if next_tok.startswith("velocity_"):
                    vel_bin = int(next_tok.split("_")[1])
                    i += 1  # consume velocity token

            # bin -> MIDI velocity (midpoint of bin range)
           
            midi_vel = max(1, vel_bin)
            active_notes[pitch] = (current_time, midi_vel)
            i += 1

        elif tok.startswith("note_off_"):
            pitch = int(tok.split("_")[2])
            if pitch in active_notes:
                onset, vel = active_notes.pop(pitch)
                notes.append((onset, current_time, pitch, vel))
            i += 1

        elif tok.startswith("velocity_"):
            # velocity tokens should be consumed inside note_on
            # if we see one here it's orphaned: skip it
            _log(f"Orphaned velocity token at position {i}, skipping")
            i += 1

        else:
            _log(f"Unknown token '{tok}' at position {i}, skipping")
            i += 1

    # Force-close remaining notes at the actual segment boundary,
    # not at the last time token (which may be earlier than the segment end).
    close_time = segment_duration if segment_duration is not None else current_time
    for pitch, (onset, vel) in active_notes.items():
        if close_time > onset:
            notes.append((onset, close_time, pitch, vel))

    _log(f"Decoded {len(notes)} notes")
    return notes

def notes_to_mir_eval(notes, log=None):

    def _log(msg):
        if log:
            log(f"    [notes_to_mir_eval] {msg}")

    if len(notes) == 0:
        _log("Empty note list")
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0,      dtype=np.float64),
            np.zeros(0,      dtype=np.float64),
        )

    notes      = sorted(notes, key=lambda x: x[0])
    intervals  = np.array([[n[0], n[1]] for n in notes], dtype=np.float64)
    pitches    = np.array([n[2]         for n in notes], dtype=np.float64)
    velocities = np.array([n[3]         for n in notes], dtype=np.float64)

    intervals[:, 1] = np.maximum(intervals[:, 1], intervals[:, 0] + 0.01)

    if len(pitches) > 0:
        _log(f"intervals shape: {intervals.shape}, pitch range: {pitches.min():.0f}-{pitches.max():.0f}")
    else:
        _log("intervals shape: (0,2), empty pitch list")
    return intervals, pitches, velocities


# Ground truth loaders


def get_gt_from_hdf5(dataset, song_idx, log=None):

    def _log(msg):
        if log:
            log(f"  [gt_hdf5] {msg}")

    _log(f"Loading GT for song {song_idx}")
    f        = dataset._get_file()
    ev_start = int(dataset.event_offsets[song_idx])
    ev_len   = int(dataset.event_lengths[song_idx])
    _log(f"event_offsets[{song_idx}]={ev_start}, event_lengths[{song_idx}]={ev_len}")

    arr = f["events"][ev_start : ev_start + ev_len]
    _log(f"events array shape: {arr.shape}, dtype: {arr.dtype}")

    notes = []
    for row in arr:
        notes.append((
            float(row[0]),
            float(row[1]),
            int(round(float(row[2]))),
            int(round(float(row[3]))),
        ))

    _log(f"Loaded {len(notes)} GT notes")
    return notes


def get_gt_from_midi(midi_path, log=None):

    def _log(msg):
        if log:
            log(f"  [gt_midi] {msg}")

    _log(f"Loading MIDI: {midi_path}")
    try:
        import pretty_midi
    except ImportError:
        print("pretty_midi not installed. Run: pip install pretty_midi --break-system-packages")
        sys.exit(1)

    pm    = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))

    _log(f"Loaded {len(notes)} notes from MIDI")
    return notes


# Per-song inference

def deduplicate_notes(notes, time_tol=0.05):
    """
    Merge duplicated notes caused by overlapping segments.
    Keeps the longest note when duplicates are found.
    """
    if len(notes) == 0:
        return notes

    notes = sorted(notes, key=lambda x: (x[2], x[0]))  # pitch, onset
    merged = []

    for note in notes:
        onset, offset, pitch, velocity = note

        if not merged:
            merged.append(note)
            continue

        prev = merged[-1]
        p_on, p_off, p_pitch, p_vel = prev

        if pitch == p_pitch and abs(onset - p_on) < time_tol:
            # keep longer note
            if offset > p_off:
                merged[-1] = (p_on, offset, pitch, velocity)
        else:
            merged.append(note)

    return merged
    
@torch.no_grad()
def transcribe_song(
    model,
    dataset,
    song_idx,
    device,
    max_output_tokens=1024,
    overlap_ratio=0.25,   # FIX 3: reduced from 0.5 — less overlap means fewer
                          # duplicate notes and simpler deduplication, which
                          # reduces offset errors that hurt KOR.
    log=None
):
    def _log(msg):
        if log:
            log(f"  [transcribe] {msg}")

    f          = dataset._get_file()
    mel_start  = int(dataset.mel_offsets[song_idx])
    mel_len    = int(dataset.mel_lengths[song_idx])

    max_frames = dataset.max_input_frames
    hop        = dataset.hop_length
    sr         = dataset.sample_rate

    seg_dur = max_frames * hop / sr
    stride  = int(max_frames * (1 - overlap_ratio))
    stride  = max(1, stride)

    stride_time = stride * hop / sr

    _log(f"Song {song_idx}")
    _log(f"mel_len={mel_len}, max_frames={max_frames}")
    _log(f"seg_dur={seg_dur:.3f}s, stride={stride}, stride_time={stride_time:.3f}s")

    all_notes = []

    frame_pos   = 0
    time_offset = 0.0
    seg_idx     = 0

    model.eval()

    while frame_pos < mel_len:

        mel = f["mel"][mel_start + frame_pos : mel_start + frame_pos + max_frames]

        # Padding
        if mel.shape[0] < max_frames:
            pad = np.zeros((max_frames - mel.shape[0], dataset.n_mels), dtype=np.float32)
            mel = np.vstack([mel, pad])

        if mel.dtype != np.float32:
            mel = mel.astype(np.float32)

        audio_t = torch.from_numpy(mel).unsqueeze(0).to(device)

        token_ids = greedy_decode(
            model,
            audio_t,
            bos_id=dataset.bos_id,
            eos_id=dataset.eos_id,
            max_len=max_output_tokens
        )

        # FIX 1: pass seg_dur so open notes close at the true segment boundary
        seg_notes = tokens_to_notes(
            token_ids,
            dataset.token_to_id,
            segment_duration=seg_dur,
        )

        kept = 0

        for onset, offset, pitch, velocity in seg_notes:

            # Skip invalid
            if offset <= 0:
                continue

            # Allow small overflow (quantization tolerance)
            if onset > seg_dur + 0.05:
                continue

            onset_global  = onset  + time_offset
            offset_global = offset + time_offset

            # Fix invalid durations
            if offset_global <= onset_global:
                offset_global = onset_global + 0.01

            # FIX 2: lowered from 0.02 to 0.01 — 20ms was discarding real
            # staccato notes; 10ms only removes truly impossible artifacts.
            if offset_global - onset_global < 0.01:
                continue

            all_notes.append((
                onset_global,
                offset_global,
                pitch,
                velocity
            ))
            kept += 1

        _log(f"Segment {seg_idx}: kept {kept}/{len(seg_notes)} notes")

        frame_pos   += stride
        time_offset += stride_time
        seg_idx     += 1

    # Sort notes globally
    all_notes = sorted(all_notes, key=lambda x: x[0])

    # Deduplicate overlap artifacts
    all_notes = deduplicate_notes(all_notes)

    _log(f"Final notes: {len(all_notes)}")

    return all_notes



# mir_eval scoring 

def score_notes(gt_notes, pred_notes, has_velocity_tolerance,
                has_transcription_velocity, log=None):

    def _log(msg):
        if log:
            log(f"  [score] {msg}")

    _log(f"GT: {len(gt_notes)} notes, Pred: {len(pred_notes)} notes")

    gt_iv,   gt_p,   gt_v   = notes_to_mir_eval(gt_notes,   log=log)
    pred_iv, pred_p, pred_v = notes_to_mir_eval(pred_notes, log=log)

    results = {}

    # Onset only
    _log("Computing Onset F1")
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, pred_iv, pred_p,
        onset_tolerance=0.05,
        offset_ratio=None,
    )
    results["onset"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    _log(f"  Onset F1: {f*100:.2f}%")

    # Onset + Offset
    _log("Computing Onset+Offset F1")
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, pred_iv, pred_p,
        onset_tolerance=0.05,
        offset_ratio=0.2,
    )

    results["onset_offset"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    _log(f"  Onset+Offset F1: {f*100:.2f}%")

    # Onset + Offset + Velocity
    _log("Computing Onset+Offset+Velocity F1")
    try:
        if has_velocity_tolerance:
            # Newer mir_eval: velocity_tolerance kwarg in precision_recall_f1_overlap
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                gt_iv, gt_p, pred_iv, pred_p,
                onset_tolerance=0.05,
                offset_ratio=0.2,
                velocity_tolerance=0.1,
            )
        elif has_transcription_velocity:
            # Older mir_eval: separate transcription_velocity module
            p, r, f, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
                gt_iv, gt_p, gt_v,
                pred_iv, pred_p, pred_v,
                onset_tolerance=0.05,
                offset_ratio=0.2,
                velocity_tolerance=0.1,
            )
        else:
            _log(" WARNING: mir_eval version has no velocity support, using onset+offset as proxy")
            p = results["onset_offset"]["precision"]
            r = results["onset_offset"]["recall"]
            f = results["onset_offset"]["f1"]

    except Exception as e:
        _log(f"  WARNING: velocity scoring failed ({e}), using onset+offset as proxy")
        p = results["onset_offset"]["precision"]
        r = results["onset_offset"]["recall"]
        f = results["onset_offset"]["f1"]

    results["onset_offset_vel"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    _log(f"  Onset+Offset+Vel F1: {f*100:.2f}%")

    return results



def notes_to_midi(notes, output_path):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for onset, offset, pitch, velocity in notes:
        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(onset),
            end=float(offset)
        )
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_path)


def compute_mpteval_metrics(gt_midi_path, pred_midi_path, log=None):

    def _log(msg):
        if log: log(f"  [mpteval] {msg}")

    try:
        import partitura as pt
        from mpteval.timing       import timing_metrics_from_perf
        from mpteval.dynamics     import dynamics_metrics_from_perf
        from mpteval.articulation import articulation_metrics_from_perf

        ref_perf  = pt.load_performance_midi(gt_midi_path)
        pred_perf = pt.load_performance_midi(pred_midi_path)

        timing       = timing_metrics_from_perf(ref_perf, pred_perf)
        dynamics     = dynamics_metrics_from_perf(ref_perf, pred_perf)
        articulation = articulation_metrics_from_perf(ref_perf, pred_perf)

        #_log(f"raw timing:       {timing}")
        #_log(f"raw dynamics:     {dynamics}")
        #_log(f"raw articulation: {articulation}")

        # timing: ndarray of shape (1, 3), (melody_ioi, accompaniment_ioi, ?)
        t = timing[0]
        melody_ioi        = float(t[0])
        accompaniment_ioi = float(t[1])

        # dynamics: single float
        dyn = float(dynamics)

        # articulation: ndarray of shape (2, 4), two rows for vel threshold 64 and 127
        # (melody_kor, bass_kor, ratio_kor, ?)
        # use row 0 (threshold=64) as the main metric, row 1 (threshold=127) as strict
        a0 = articulation[0]
        a1 = articulation[1]
        melody_kor        = float(a0[0])
        bass_kor          = float(a0[1])
        ratio_kor         = float(a0[2])
        melody_kor_strict = float(a1[0])
        bass_kor_strict   = float(a1[1])
        ratio_kor_strict  = float(a1[2])

        result = {
            "melody_ioi":        melody_ioi,
            "accompaniment_ioi": accompaniment_ioi,
            "dynamics":          dyn,
            "melody_kor":        melody_kor,
            "bass_kor":          bass_kor,
            "ratio_kor":         ratio_kor,
            "melody_kor_strict": melody_kor_strict,
            "bass_kor_strict":   bass_kor_strict,
            "ratio_kor_strict":  ratio_kor_strict,
        }

        _log(f"Parsed metrics: {result}")
        return result

    except Exception as e:
        _log(f"WARNING: MPTEval failed: {e}")
        _log(traceback.format_exc())
        return {}


def save_results(per_song_results, all_scores, failed, step, midi_files, results_path, log):
    n_eval = len(per_song_results)
    if n_eval == 0:
        return

    def mean_pct(key, metric):
        return float(np.mean(all_scores[key][metric])) * 100 if all_scores[key][metric] else 0.0

    mpteval_keys = ["melody_ioi", "accompaniment_ioi", "dynamics",
                    "melody_kor", "bass_kor", "ratio_kor"]
    mpteval_summary = {}
    for key in mpteval_keys:
        vals = [s[key] for s in per_song_results if s.get(key) is not None]
        mpteval_summary[key] = float(np.mean(vals)) if vals else None

    results = {
        "checkpoint":       args.checkpoint,
        "step":             step,
        "num_evaluated":    n_eval,
        "num_failed":       failed,
        "gt_source":        "midi" if midi_files else "hdf5",
        "mir_eval_version": mir_eval.__version__,
        "summary": {
            "onset":            {m: mean_pct("onset",            m) for m in ["precision","recall","f1"]},
            "onset_offset":     {m: mean_pct("onset_offset",     m) for m in ["precision","recall","f1"]},
            "onset_offset_vel": {m: mean_pct("onset_offset_vel", m) for m in ["precision","recall","f1"]},
        },
        "mpteval_summary": mpteval_summary,
        "per_song":         per_song_results,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

def evaluate(args):

    os.makedirs(args.output_dir, exist_ok=True)
    log_path     = os.path.join(args.output_dir, "eval.log")
    results_path = os.path.join(args.output_dir, "results.json")

    def log(msg):
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    
    log("AMT EVALUATION")
    log(f"Checkpoint:  {args.checkpoint}")
    log(f"Test HDF5:   {args.hdf5_test}")
    log(f"MIDI dir:    {args.midi_dir or 'not provided — using HDF5 events'}")
    log(f"Output dir:  {args.output_dir}")
    log(f"Num samples: {args.num_samples if args.num_samples else 'ALL'}")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, VRAM: {props.total_memory/1e9:.1f}GB")

    has_vel_tol, has_trans_vel = check_mir_eval_version(log)

    # Dataset

    log("Loading test dataset")
    try:
        dataset = MAESTROSeq2SeqDataset(
            hdf5_path=args.hdf5_test,
            split="test",
            max_output_tokens=1024,
        )
        log(f"Dataset loaded: {len(dataset)} songs")
        log(f"Vocab size: {len(dataset.vocab)}")
        log(f"bos_id={dataset.bos_id}, eos_id={dataset.eos_id}, pad_id={dataset.pad_id}")
    except Exception as e:
        log(f"FATAL: Dataset load failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    num_songs = len(dataset)
    if args.num_samples:
        num_songs = min(args.num_samples, num_songs)
    log(f"Will evaluate {num_songs} songs")

    # MIDI index

    midi_files = None
    if args.midi_dir:
        log(f"Scanning MIDI dir: {args.midi_dir}")
        midi_files = sorted([
            os.path.join(args.midi_dir, fn)
            for fn in os.listdir(args.midi_dir)
            if fn.endswith(".mid") or fn.endswith(".midi")
        ])
        log(f"Found {len(midi_files)} MIDI files")
        if len(midi_files) < num_songs:
            log(f"WARNING: fewer MIDI files than songs, capping at {len(midi_files)}")
            num_songs = min(num_songs, len(midi_files))

    # Model

    log(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        log(f"Checkpoint keys: {list(checkpoint.keys())}")
        step = checkpoint.get("step", "unknown")
        log(f"Checkpoint step: {step}")
    except Exception as e:
        log(f"FATAL: Checkpoint load failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    log("Creating model")
    try:
        model = AudioTransformer(n_mels=512, vocab_size=len(dataset.vocab))
        log(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    except Exception as e:
        log(f"FATAL: Model creation failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    log("Loading state dict")
    try:
        state_dict = checkpoint["model_state_dict"]
        has_module = any(k.startswith("module.") for k in state_dict.keys())
        log(f"State dict has 'module.' prefix: {has_module}")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        log("State dict loaded OK")
    except Exception as e:
        log(f"FATAL: state_dict load failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    model = model.to(device)
    model.eval()
    log(f"Model on {device}")

    log("Testing dummy forward pass")
    try:
        with torch.no_grad():
            dummy_out = model(
                torch.zeros(1, 512, 512).to(device),
                torch.zeros(1, 10, dtype=torch.long).to(device)
            )
            log(f"Dummy forward pass OK, output: {dummy_out.shape}")
    except Exception as e:
        log(f"FATAL: Dummy forward pass failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    # Evaluation loop
    
    # Resume from existing results if available
    per_song_results = []
    all_scores = defaultdict_scores()
    failed = 0

    if os.path.exists(results_path):
        log(f"Found existing results at {results_path}, attempting to resume")
        try:
            with open(results_path, "r") as f:
                existing = json.load(f)
            per_song_results = existing.get("per_song", [])
            failed = existing.get("num_failed", 0)

            for s in per_song_results:
                all_scores["onset"]["f1"].append(s["onset_f1"])
                all_scores["onset"]["precision"].append(s["onset_precision"])
                all_scores["onset"]["recall"].append(s["onset_recall"])
                all_scores["onset_offset"]["f1"].append(s["onset_offset_f1"])
                all_scores["onset_offset"]["precision"].append(s["onset_offset_precision"])
                all_scores["onset_offset"]["recall"].append(s["onset_offset_recall"])
                all_scores["onset_offset_vel"]["f1"].append(s["onset_offset_vel_f1"])
                all_scores["onset_offset_vel"]["precision"].append(s["onset_offset_vel_precision"])
                all_scores["onset_offset_vel"]["recall"].append(s["onset_offset_vel_recall"])

            already_done = {s["song_idx"] for s in per_song_results}
            start_song = max(already_done) + 1 if already_done else 0
            log(f"Resuming from song {start_song} ({len(per_song_results)} already evaluated)")
        except Exception as e:
            log(f"WARNING: Could not load existing results ({e}), starting fresh")
            per_song_results = []
            all_scores = defaultdict_scores()
            start_song = 0
    else:
        start_song = 0

    for song_idx in range(start_song, num_songs):

        log(f"Song {song_idx+1}/{num_songs} (idx={song_idx})")
        t0 = time.time()

        try:
            pred_notes = transcribe_song(
                model, dataset, song_idx, device,
                max_output_tokens=1024, log=log
            )
            log(f"  Predicted: {len(pred_notes)} notes")

            if midi_files is not None:
                gt_notes = get_gt_from_midi(midi_files[song_idx], log=log)
            else:
                gt_notes = get_gt_from_hdf5(dataset, song_idx, log=log)
            log(f"  GT: {len(gt_notes)} notes")

            pred_midi_path = f"{args.output_dir}/pred_{song_idx}.mid"
            gt_midi_path = f"{args.output_dir}/gt_{song_idx}.mid"

            notes_to_midi(pred_notes, pred_midi_path)
            notes_to_midi(gt_notes, gt_midi_path)
            
            mpteval_scores = compute_mpteval_metrics(
                gt_midi_path,
                pred_midi_path,
                log=log
            )
            try:
                os.remove(pred_midi_path)
                os.remove(gt_midi_path)
            except Exception:
                pass

            scores = score_notes(
                gt_notes, pred_notes,
                has_vel_tol, has_trans_vel,
                log=log
            )

        except Exception as e:
            log(f"  FAILED: {e}")
            log(traceback.format_exc())
            failed += 1
            continue

        for metric, vals in scores.items():
            all_scores[metric]["f1"].append(vals["f1"])
            all_scores[metric]["precision"].append(vals["precision"])
            all_scores[metric]["recall"].append(vals["recall"])

        elapsed = time.time() - t0
        log(
            f"  DONE {elapsed:.1f}s | "
            f"Onset F1: {scores['onset']['f1']*100:.1f}% | "
            f"O+Off F1: {scores['onset_offset']['f1']*100:.1f}% | "
            f"O+Off+V F1: {scores['onset_offset_vel']['f1']*100:.1f}%"
        )

        per_song_results.append({
            "song_idx": song_idx,
            "gt_notes": len(gt_notes),
            "pred_notes": len(pred_notes),

            "onset_f1": scores["onset"]["f1"],
            "onset_precision": scores["onset"]["precision"],
            "onset_recall": scores["onset"]["recall"],

            "onset_offset_f1": scores["onset_offset"]["f1"],
            "onset_offset_precision": scores["onset_offset"]["precision"],
            "onset_offset_recall": scores["onset_offset"]["recall"],

            "onset_offset_vel_f1": scores["onset_offset_vel"]["f1"],
            "onset_offset_vel_precision": scores["onset_offset_vel"]["precision"],
            "onset_offset_vel_recall": scores["onset_offset_vel"]["recall"],

            "melody_ioi":        mpteval_scores.get("melody_ioi",        None),
            "accompaniment_ioi": mpteval_scores.get("accompaniment_ioi", None),
            "melody_kor":        mpteval_scores.get("melody_kor",        None),
            "bass_kor":          mpteval_scores.get("bass_kor",          None),
            "ratio_kor":         mpteval_scores.get("ratio_kor",         None),
            "dynamics":          mpteval_scores.get("dynamics",          None)
        })
        save_results(per_song_results, all_scores, failed, step,
                     midi_files, results_path, log)

    n_eval = len(per_song_results)
    if n_eval == 0:
        log("ERROR: No songs evaluated successfully")
        sys.exit(1)

    save_results(per_song_results, all_scores, failed, step,
                midi_files, results_path, log)

    log(f"RESULTS  ({n_eval} evaluated, {failed} failed)")
    log(f"Results saved to {results_path}")
    log(f"Log saved to     {log_path}")


# Helper

def defaultdict_scores():
    return {
        "onset":            {"f1": [], "precision": [], "recall": []},
        "onset_offset":     {"f1": [], "precision": [], "recall": []},
        "onset_offset_vel": {"f1": [], "precision": [], "recall": []},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--hdf5_test",    type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--midi_dir",    type=str, default=None)
    args = parser.parse_args()

    try:
        evaluate(args)
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)