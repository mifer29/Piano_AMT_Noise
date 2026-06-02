"""

Piano AMT model evaluation across multiple noise conditions using mir_eval and mpteval.
Loops over all conditions stored in the test HDF5 (clean, noise_10db, etc.)
and writes one results_{condition}.json per condition to output_dir.

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

from maestro_dataset import MAESTROSeq2SeqDataset
from model import AudioTransformer, greedy_decode

import partitura as pt
from mpteval.timing       import timing_metrics_from_perf
from mpteval.dynamics     import dynamics_metrics_from_perf
from mpteval.articulation import articulation_metrics_from_perf



# mir_eval version detection


def check_mir_eval_version(log):
    log(f"mir_eval version: {mir_eval.__version__}")
    has_velocity_tolerance = "velocity_tolerance" in str(
        mir_eval.transcription.precision_recall_f1_overlap.__doc__ or ""
    )
    has_transcription_velocity = hasattr(mir_eval, "transcription_velocity")
    log(f"  has velocity_tolerance kwarg: {has_velocity_tolerance}")
    log(f"  has transcription_velocity module: {has_transcription_velocity}")
    return has_velocity_tolerance, has_transcription_velocity


# Token to note conversion

def tokens_to_notes(token_ids, token_to_id, time_resolution=0.01,
                    segment_duration=None, log=None):
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

        elif tok.startswith("velocity_"):
            _log(f"Orphaned velocity token at position {i}, skipping")
            i += 1

        else:
            _log(f"Unknown token '{tok}' at position {i}, skipping")
            i += 1

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
        _log(f"intervals shape: {intervals.shape}, "
             f"pitch range: {pitches.min():.0f}-{pitches.max():.0f}")
    return intervals, pitches, velocities



# Ground truth loaders

def get_gt_from_hdf5(dataset, song_idx, log=None):
    def _log(msg):
        if log:
            log(f"  [gt_hdf5] {msg}")

    hdf5_file = dataset._get_file()
    ev_start  = int(dataset.event_offsets[song_idx])
    ev_len    = int(dataset.event_lengths[song_idx])
    arr       = hdf5_file["events"][ev_start : ev_start + ev_len]

    notes = [
        (float(r[0]), float(r[1]), int(round(float(r[2]))), int(round(float(r[3]))))
        for r in arr
    ]
    _log(f"Loaded {len(notes)} GT notes")
    return notes


def get_gt_from_midi(midi_path, log=None):
    def _log(msg):
        if log:
            log(f"  [gt_midi] {msg}")

    pm    = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))

    _log(f"Loaded {len(notes)} notes from MIDI")
    return notes



# Inference


def deduplicate_notes(notes, time_tol=0.05):
    if len(notes) == 0:
        return notes

    notes  = sorted(notes, key=lambda x: (x[2], x[0]))
    merged = []

    for note in notes:
        onset, offset, pitch, velocity = note
        if not merged:
            merged.append(note)
            continue
        p_on, p_off, p_pitch, p_vel = merged[-1]
        if pitch == p_pitch and abs(onset - p_on) < time_tol:
            if offset > p_off:
                merged[-1] = (p_on, offset, pitch, velocity)
        else:
            merged.append(note)

    return merged


@torch.no_grad()
def transcribe_song(model, dataset, song_idx, device, condition,
                    max_output_tokens=1024, overlap_ratio=0.25, log=None):
    def _log(msg):
        if log:
            log(f"  [transcribe] {msg}")

    # Using a local reference to the HDF5 file: never reuse an outer variable
    # called 'f' that might be overwritten by a file context manager elsewhere.
    hdf5_file  = dataset._get_file()
    mel_start  = int(dataset.mel_offsets[song_idx])
    mel_len    = int(dataset.mel_lengths[song_idx])

    max_frames = dataset.max_input_frames
    hop        = dataset.hop_length
    sr         = dataset.sample_rate

    seg_dur     = max_frames * hop / sr
    stride      = max(1, int(max_frames * (1 - overlap_ratio)))
    stride_time = stride * hop / sr

    # The condition-specific dataset key ("mel_clean", "mel_noise_10db"...)
    mel_key = f"mel_{condition}"
    if mel_key not in hdf5_file:
        raise KeyError(
            f"Dataset '{mel_key}' not found in HDF5. "
            f"Available keys: {list(hdf5_file.keys())}"
        )

    _log(f"Song {song_idx} | condition={condition} | key={mel_key}")
    _log(f"mel_len={mel_len}, max_frames={max_frames}, "
         f"seg_dur={seg_dur:.3f}s, stride={stride}")

    all_notes   = []
    frame_pos   = 0
    time_offset = 0.0
    seg_idx     = 0

    model.eval()

    while frame_pos < mel_len:
        mel = hdf5_file[mel_key][
            mel_start + frame_pos : mel_start + frame_pos + max_frames
        ]

        if mel.shape[0] < max_frames:
            pad = np.zeros(
                (max_frames - mel.shape[0], dataset.n_mels), dtype=np.float32
            )
            mel = np.vstack([mel, pad])

        if mel.dtype != np.float32:
            mel = mel.astype(np.float32)

        audio_t   = torch.from_numpy(mel).unsqueeze(0).to(device)
        token_ids = greedy_decode(
            model, audio_t,
            bos_id=dataset.bos_id,
            eos_id=dataset.eos_id,
            max_len=max_output_tokens,
        )

        seg_notes = tokens_to_notes(
            token_ids,
            dataset.token_to_id,
            segment_duration=seg_dur,
            log=log,
        )

        kept = 0
        for onset, offset, pitch, velocity in seg_notes:
            if offset <= 0 or onset > seg_dur + 0.05:
                continue
            onset_g  = onset  + time_offset
            offset_g = max(offset + time_offset, onset_g + 0.01)
            if offset_g - onset_g < 0.01:
                continue
            all_notes.append((onset_g, offset_g, pitch, velocity))
            kept += 1

        _log(f"Segment {seg_idx}: kept {kept}/{len(seg_notes)} notes")

        frame_pos   += stride
        time_offset += stride_time
        seg_idx     += 1

    all_notes = deduplicate_notes(sorted(all_notes, key=lambda x: x[0]))
    _log(f"Final notes: {len(all_notes)}")
    return all_notes



# Scoring

def score_notes(gt_notes, pred_notes, has_velocity_tolerance,
                has_transcription_velocity, log=None):
    def _log(msg):
        if log:
            log(f"  [score] {msg}")

    _log(f"GT: {len(gt_notes)} notes, Pred: {len(pred_notes)} notes")

    gt_iv,   gt_p,   gt_v   = notes_to_mir_eval(gt_notes,   log=log)
    pred_iv, pred_p, pred_v = notes_to_mir_eval(pred_notes, log=log)

    results = {}

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, pred_iv, pred_p,
        onset_tolerance=0.05, offset_ratio=None,
    )
    results["onset"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    _log(f"  Onset F1: {f*100:.2f}%")

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_p, pred_iv, pred_p,
        onset_tolerance=0.05, offset_ratio=0.2,
    )
    results["onset_offset"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    _log(f"  Onset+Offset F1: {f*100:.2f}%")

    try:
        if has_velocity_tolerance:
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                gt_iv, gt_p, pred_iv, pred_p,
                onset_tolerance=0.05, offset_ratio=0.2, velocity_tolerance=0.1,
            )
        elif has_transcription_velocity:
            p, r, f, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
                gt_iv, gt_p, gt_v, pred_iv, pred_p, pred_v,
                onset_tolerance=0.05, offset_ratio=0.2, velocity_tolerance=0.1,
            )
        else:
            _log("WARNING: no velocity support in mir_eval, using onset+offset")
            p = results["onset_offset"]["precision"]
            r = results["onset_offset"]["recall"]
            f = results["onset_offset"]["f1"]
    except Exception as e:
        _log(f"WARNING: velocity scoring failed ({e}), using onset+offset")
        p = results["onset_offset"]["precision"]
        r = results["onset_offset"]["recall"]
        f = results["onset_offset"]["f1"]

    results["onset_offset_vel"] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    _log(f"  Onset+Offset+Vel F1: {f*100:.2f}%")

    return results



# MIDI helpers

def notes_to_midi(notes, output_path):
    pm         = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for onset, offset, pitch, velocity in notes:
        instrument.notes.append(pretty_midi.Note(
            velocity=int(np.clip(velocity, 1, 127)),
            pitch=int(pitch),
            start=float(onset),
            end=float(offset),
        ))
    pm.instruments.append(instrument)
    pm.write(output_path)


def compute_mpteval_metrics(gt_midi_path, pred_midi_path, log=None):
    def _log(msg):
        if log:
            log(f"  [mpteval] {msg}")

    try:

        ref_perf  = pt.load_performance_midi(gt_midi_path)
        pred_perf = pt.load_performance_midi(pred_midi_path)

        timing       = timing_metrics_from_perf(ref_perf, pred_perf)
        dynamics     = dynamics_metrics_from_perf(ref_perf, pred_perf)
        articulation = articulation_metrics_from_perf(ref_perf, pred_perf)

        t  = timing[0]
        a0 = articulation[0]
        a1 = articulation[1]

        result = {
            "melody_ioi":        float(t[0]),
            "accompaniment_ioi": float(t[1]),
            "dynamics":          float(dynamics),
            "melody_kor":        float(a0[0]),
            "bass_kor":          float(a0[1]),
            "ratio_kor":         float(a0[2]),
            "melody_kor_strict": float(a1[0]),
            "bass_kor_strict":   float(a1[1]),
            "ratio_kor_strict":  float(a1[2]),
        }

        _log(f"Parsed metrics: {result}")
        return result

    except Exception as e:
        # Log the full traceback so you can see the real failure reason
        _log(f" MPTEval failed: {e}")
        _log(traceback.format_exc())
        return {}



# Results saving


def defaultdict_scores():
    return {
        "onset":            {"f1": [], "precision": [], "recall": []},
        "onset_offset":     {"f1": [], "precision": [], "recall": []},
        "onset_offset_vel": {"f1": [], "precision": [], "recall": []},
    }


def save_results(per_song_results, all_scores, failed, step,
                 midi_files, results_path, args, log):
    """
    Saves results to JSON. Uses a named argument for args to avoid
    relying on a global — prevents the 'f' variable shadowing issue.
    """
    n_eval = len(per_song_results)
    if n_eval == 0:
        return

    def mean_pct(key, metric):
        vals = all_scores[key][metric]
        return float(np.mean(vals)) * 100 if vals else 0.0

    mpteval_keys = [
        "melody_ioi", "accompaniment_ioi", "dynamics",
        "melody_kor", "bass_kor", "ratio_kor",
        "melody_kor_strict", "bass_kor_strict", "ratio_kor_strict",
    ]
    mpteval_summary = {}
    for key in mpteval_keys:
        vals = [s[key] for s in per_song_results if s.get(key) is not None]
        mpteval_summary[key] = float(np.mean(vals)) if vals else None

    output = {
        "checkpoint":       args.checkpoint,
        "step":             step,
        "num_evaluated":    n_eval,
        "num_failed":       failed,
        "gt_source":        "midi" if midi_files else "hdf5",
        "mir_eval_version": mir_eval.__version__,
        "summary": {
            "onset":            {m: mean_pct("onset",            m) for m in ["precision", "recall", "f1"]},
            "onset_offset":     {m: mean_pct("onset_offset",     m) for m in ["precision", "recall", "f1"]},
            "onset_offset_vel": {m: mean_pct("onset_offset_vel", m) for m in ["precision", "recall", "f1"]},
        },
        "mpteval_summary": mpteval_summary,
        "per_song":         per_song_results,
    }

    # Write to a temp file first then rename: avoids corrupt JSON on crash
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(output, fh, indent=2)
    os.replace(tmp_path, results_path)



# Main evaluation loop


def evaluate(args):

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "eval.log")

    def log(msg):
        ts   = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:   # 'fh' not 'f' — avoids shadowing HDF5 handle
            fh.write(line + "\n")

    log("AMT NOISE EVALUATION")
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
    dataset = MAESTROSeq2SeqDataset(
        hdf5_path=args.hdf5_test,
        split="test",
        max_output_tokens=1024,
    )
    log(f"Dataset loaded: {len(dataset)} songs")

    num_songs = len(dataset)
    if args.num_samples:
        num_songs = min(args.num_samples, num_songs)

    # MIDI index 

    midi_files = None
    if args.midi_dir:
        midi_files = sorted([
            os.path.join(args.midi_dir, fn)
            for fn in os.listdir(args.midi_dir)
            if fn.endswith(".mid") or fn.endswith(".midi")
        ])
        num_songs = min(num_songs, len(midi_files))
        log(f"Using {len(midi_files)} MIDI files for GT")

    # Model

    log(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    step       = checkpoint.get("step", "unknown")
    log(f"Checkpoint step: {step}")

    model = AudioTransformer(n_mels=512, vocab_size=len(dataset.vocab))
    state_dict = {
        k.replace("module.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    log(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Discover conditions from HDF5 attrs 

    hdf5_handle = dataset._get_file()
    if "conditions" in hdf5_handle.attrs:
        conditions = list(hdf5_handle.attrs["conditions"])
    else:
        # Fallback: infer from dataset keys named mel_*
        conditions = [
            k.replace("mel_", "")
            for k in hdf5_handle.keys()
            if k.startswith("mel_")
        ]
    log(f"Conditions found: {conditions}")

    # Loop over conditions

    for condition in conditions:

        log(f"\n{'='*60}")
        log(f"CONDITION: {condition}")
        log(f"{'='*60}")

        results_path = os.path.join(args.output_dir, f"results_{condition}.json")

        # Resume
        per_song_results = []
        all_scores       = defaultdict_scores()
        failed           = 0
        start_song       = 0

        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as fh:
                    existing = json.load(fh)
                per_song_results = existing.get("per_song", [])
                failed           = existing.get("num_failed", 0)
                for s in per_song_results:
                    for metric in ["onset", "onset_offset", "onset_offset_vel"]:
                        all_scores[metric]["f1"].append(s[f"{metric}_f1"])
                        all_scores[metric]["precision"].append(s[f"{metric}_precision"])
                        all_scores[metric]["recall"].append(s[f"{metric}_recall"])
                start_song = max((s["song_idx"] for s in per_song_results), default=-1) + 1
                log(f"Resuming from song {start_song} "
                    f"({len(per_song_results)} already evaluated)")
            except Exception as e:
                log(f" Could not resume ({e}), starting fresh")
                per_song_results = []
                all_scores       = defaultdict_scores()
                start_song       = 0

        # Per-song loop

        for song_idx in range(start_song, num_songs):

            log(f"[{condition}] Song {song_idx + 1}/{num_songs} (idx={song_idx})")
            t0 = time.time()

            try:
                pred_notes = transcribe_song(
                    model, dataset, song_idx, device,
                    condition=condition,
                    max_output_tokens=1024,
                    log=log,
                )
                log(f"  Predicted: {len(pred_notes)} notes")

                gt_notes = (
                    get_gt_from_midi(midi_files[song_idx], log=log)
                    if midi_files else
                    get_gt_from_hdf5(dataset, song_idx, log=log)
                )
                log(f"  GT: {len(gt_notes)} notes")

                pred_midi_path = os.path.join(
                    args.output_dir, f"pred_{condition}_{song_idx}.mid"
                )
                gt_midi_path = os.path.join(
                    args.output_dir, f"gt_{condition}_{song_idx}.mid"
                )

                notes_to_midi(pred_notes, pred_midi_path)
                notes_to_midi(gt_notes,   gt_midi_path)

                mpteval_scores = compute_mpteval_metrics(
                    gt_midi_path, pred_midi_path, log=log
                )

                try:
                    os.remove(pred_midi_path)
                    os.remove(gt_midi_path)
                except Exception:
                    pass

                scores = score_notes(
                    gt_notes, pred_notes,
                    has_vel_tol, has_trans_vel,
                    log=log,
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

            # Store the COMPLETE per-song dict: same fields as evaluate.py
            # so save_results can compute the mpteval_summary without nulls
            per_song_results.append({
                "song_idx":  song_idx,
                "condition": condition,
                "gt_notes":  len(gt_notes),
                "pred_notes": len(pred_notes),

                "onset_f1":        scores["onset"]["f1"],
                "onset_precision": scores["onset"]["precision"],
                "onset_recall":    scores["onset"]["recall"],

                "onset_offset_f1":        scores["onset_offset"]["f1"],
                "onset_offset_precision": scores["onset_offset"]["precision"],
                "onset_offset_recall":    scores["onset_offset"]["recall"],

                "onset_offset_vel_f1":        scores["onset_offset_vel"]["f1"],
                "onset_offset_vel_precision": scores["onset_offset_vel"]["precision"],
                "onset_offset_vel_recall":    scores["onset_offset_vel"]["recall"],

                "melody_ioi":        mpteval_scores.get("melody_ioi"),
                "accompaniment_ioi": mpteval_scores.get("accompaniment_ioi"),
                "dynamics":          mpteval_scores.get("dynamics"),
                "melody_kor":        mpteval_scores.get("melody_kor"),
                "bass_kor":          mpteval_scores.get("bass_kor"),
                "ratio_kor":         mpteval_scores.get("ratio_kor"),
                "melody_kor_strict": mpteval_scores.get("melody_kor_strict"),
                "bass_kor_strict":   mpteval_scores.get("bass_kor_strict"),
                "ratio_kor_strict":  mpteval_scores.get("ratio_kor_strict"),
            })

            save_results(
                per_song_results, all_scores, failed, step,
                midi_files, results_path, args, log,
            )

            log(
                f"  DONE {time.time()-t0:.1f}s | "
                f"Onset F1: {scores['onset']['f1']*100:.1f}% | "
                f"O+Off F1: {scores['onset_offset']['f1']*100:.1f}% | "
                f"KOR: {mpteval_scores.get('melody_kor', float('nan')):.3f}"
            )

        log(f"Finished condition: {condition} "
            f"({len(per_song_results)} songs, {failed} failed)")

    log("\nAll conditions complete.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AMT model across multiple noise conditions."
    )
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--hdf5_test",   type=str, required=True)
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