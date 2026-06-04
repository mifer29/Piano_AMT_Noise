import logging
import os
import subprocess
import shutil
import cv2
import numpy as np

from config import (
    SOUNDFONT_PATH, FFMPEG_TIMEOUT, FLUIDSYNTH_TIMEOUT, FFPROBE_TIMEOUT,
    FPS, WIDTH, HEIGHT, KEYBOARD_H, SPEED, LEAD_IN, LEAD_OUT, SPLIT_PITCH,
    MIDI_MIN, MIDI_MAX, WHITE_OFFSETS, BLACK_OFFSETS, NUM_WHITE_KEYS,
    BG_COLOR, WHITE_KEY_COLOR, BLACK_KEY_COLOR, NOTE_COLOR_RIGHT, NOTE_COLOR_LEFT,
    NOTE_BORDER, HIT_LINE_COLOR, DIVIDER_COLOR,
)

log = logging.getLogger("transcribe")



# Piano keyboard layout

def build_key_layout(frame_w, keyboard_h):
    white_w = frame_w / NUM_WHITE_KEYS
    black_w = white_w * 0.58
    black_h = keyboard_h * 0.62
    layout  = {}
    white_index = 0

    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        semitone = midi % 12
        is_black = semitone in BLACK_OFFSETS
        if not is_black:
            layout[midi] = (white_index * white_w, white_w, False, keyboard_h)
            white_index += 1
        else:
            prev_white = midi - 1
            while (prev_white % 12) not in WHITE_OFFSETS:
                prev_white -= 1
            wi = sum(1 for m in range(MIDI_MIN, prev_white + 1) if (m % 12) in WHITE_OFFSETS) - 1
            x  = (wi + 1) * white_w - black_w / 2
            layout[midi] = (x, black_w, True, black_h)

    return layout, white_w



# Ghost note filter

def remove_octave_ghosts(notes, time_tol=0.05, vel_diff_min=8, min_duration=0.040):
    to_remove = set()
    n = len(notes)
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            a, b = notes[i], notes[j]
            if b.start - a.start > time_tol:
                break
            if a.pitch % 12 != b.pitch % 12:
                continue
            if abs(a.pitch - b.pitch) != 12:
                continue
            if abs(int(a.velocity) - int(b.velocity)) < vel_diff_min:
                continue
            if a.velocity >= b.velocity:
                to_remove.add(j)
            else:
                to_remove.add(i)
                break

    after_pass1 = [note for idx, note in enumerate(notes) if idx not in to_remove]
    return [note for note in after_pass1 if (note.end - note.start) >= min_duration]



# Frame renderer

def velocity_alpha(velocity, min_a=0.55, max_a=1.0):
    return min_a + (max_a - min_a) * (velocity / 127.0)

def render_frame(frame, notes, current_time, layout, frame_w, frame_h, keyboard_h):
    hit_y = frame_h - keyboard_h
    frame[:] = BG_COLOR

    # Falling notes
    for note in notes:
        if note.pitch < MIDI_MIN or note.pitch > MIDI_MAX:
            continue
        note_bottom_y = hit_y - (note.start - current_time) * SPEED
        note_top_y    = hit_y - (note.end   - current_time) * SPEED
        if note_top_y > hit_y or note_bottom_y < 0:
            continue
        draw_top    = max(int(note_top_y),    0)
        draw_bottom = min(int(note_bottom_y), hit_y)
        if draw_bottom <= draw_top:
            continue
        x_left, key_w, is_black, _ = layout[note.pitch]
        x1    = int(x_left) + 1
        x2    = int(x_left + key_w) - 1
        color = NOTE_COLOR_RIGHT if note.pitch >= SPLIT_PITCH else NOTE_COLOR_LEFT
        alpha = velocity_alpha(note.velocity)
        roi   = frame[draw_top:draw_bottom, x1:x2]
        note_rect = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(note_rect, alpha, roi, 1 - alpha, 0, roi)
        frame[draw_top:draw_bottom, x1:x2] = roi
        cv2.rectangle(frame, (x1, draw_top), (x2, draw_bottom), NOTE_BORDER, 1)
        if draw_top > 0:
            cv2.line(frame, (x1, draw_top), (x2, draw_top), NOTE_BORDER, 2)

    # Hit line
    cv2.line(frame, (0, hit_y), (frame_w, hit_y), HIT_LINE_COLOR, 2)

    # Active keys
    active_keys = {
        note.pitch: (NOTE_COLOR_RIGHT if note.pitch >= SPLIT_PITCH else NOTE_COLOR_LEFT)
        for note in notes if note.start <= current_time < note.end
    }

    # White keys
    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        if midi not in layout:
            continue
        x_left, key_w, is_black, key_h = layout[midi]
        if is_black:
            continue
        x1, x2 = int(x_left), int(x_left + key_w) - 1
        color = active_keys.get(midi, WHITE_KEY_COLOR)
        cv2.rectangle(frame, (x1, hit_y), (x2, frame_h), color, -1)
        cv2.rectangle(frame, (x1, hit_y), (x2, frame_h), DIVIDER_COLOR, 1)

    # Black keys on top
    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        if midi not in layout:
            continue
        x_left, key_w, is_black, key_h = layout[midi]
        if not is_black:
            continue
        x1, x2 = int(x_left), int(x_left + key_w)
        y2      = hit_y + int(key_h)
        color   = active_keys.get(midi, BLACK_KEY_COLOR)
        cv2.rectangle(frame, (x1, hit_y), (x2, y2), color, -1)
        cv2.line(frame,      (x1, hit_y), (x2, hit_y), (80, 80, 80), 1)



# Full video generator

def generate_video(pm_notes, midi_path, output_path):
    """
    pm_notes : list of pretty_midi.Note objects (from pretty_midi, not raw tuples)
    midi_path: path to the .mid file (for fluidsynth audio synthesis)
    output_path: final .mp4 destination
    Returns: path to the final video with audio, H.264 encoded for Android
    """

    # Step 0: filter ghosts
    pm_notes.sort(key=lambda n: n.start)
    pm_notes = remove_octave_ghosts(pm_notes)
    if not pm_notes:
        raise ValueError("No notes to render after filtering")

    # Step 1: render frames 
    t_start      = -LEAD_IN
    t_end        = max(n.end for n in pm_notes) + LEAD_OUT
    duration     = t_end - t_start
    total_frames = int(duration * FPS)

    layout, _ = build_key_layout(WIDTH, KEYBOARD_H)

    raw_video  = output_path.replace(".mp4", "_raw.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(raw_video, fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter, check opencv installation")

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(total_frames):
        current_time = t_start + i / FPS
        render_frame(frame, pm_notes, current_time, layout, WIDTH, HEIGHT, KEYBOARD_H)
        writer.write(frame)
    writer.release()
    log.info(f"[video] {total_frames} frames rendered")

    # Step 2: synthesize audio with fluidsynth 
    synth_wav  = output_path.replace(".mp4", "_synth.wav")
    padded_wav = output_path.replace(".mp4", "_padded.wav")
    final_path = output_path

    def encode_silent_fallback():
        """Re-encode the raw mp4v video to H.264 with no audio track."""
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_video,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            final_path
        ], check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        if os.path.exists(raw_video):
            os.remove(raw_video)

    if not os.path.exists(SOUNDFONT_PATH) or shutil.which("fluidsynth") is None:
        log.warning("[video] Soundfont or fluidsynth missing — producing silent video")
        encode_silent_fallback()
        return final_path

    try:
        r = subprocess.run([
            "fluidsynth", "-ni", "-g", "1.0",
            "-F", synth_wav, "-r", "44100",
            SOUNDFONT_PATH, midi_path
        ], capture_output=True, text=True, timeout=FLUIDSYNTH_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.error(f"[video] fluidsynth timed out after {FLUIDSYNTH_TIMEOUT}s — producing silent video")
        encode_silent_fallback()
        return final_path

    if r.returncode != 0:
        log.error(f"[video] fluidsynth failed:\n{r.stderr}")
        encode_silent_fallback()
        return final_path

    # Diagnostics
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", synth_wav
        ], capture_output=True, text=True, timeout=FFPROBE_TIMEOUT)
        audio_dur = float(probe.stdout.strip()) if probe.returncode == 0 else -1
    except (subprocess.TimeoutExpired, ValueError):
        audio_dur = -1
    video_dur = total_frames / FPS
    first_note_t = min(n.start for n in pm_notes)
    log.info(f"[sync] video={video_dur:.3f}s audio={audio_dur:.3f}s "
             f"lead_in={LEAD_IN}s first_note={first_note_t:.3f}s")

    # Step 3: prepend LEAD_IN seconds of silence 
    padding_failed = False
    try:
        r2 = subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-i", synth_wav,
            "-filter_complex",
            f"[0]atrim=duration={LEAD_IN}[silence];[silence][1]concat=n=2:v=0:a=1",
            padded_wav
        ], capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if r2.returncode != 0:
            log.error(f"[video] ffmpeg padding failed:\n{r2.stderr}")
            padding_failed = True
    except subprocess.TimeoutExpired:
        log.error("[video] ffmpeg padding timed out")
        padding_failed = True

    if padding_failed:
        # WARNING: audio will be ~LEAD_IN seconds out of sync with the video.
        # Better than producing no video at all, but the desync is visible.
        log.warning(f"[video] Falling back to unpadded audio, expect {LEAD_IN}s desync")
        padded_wav = synth_wav

    # Step 4: mux video + audio, re-encode to H.264 for Android 
    try:
        r3 = subprocess.run([
            "ffmpeg", "-y",
            "-i", raw_video,
            "-i", padded_wav,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-profile:v", "baseline", "-level", "3.1",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-shortest",
            final_path
        ], capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        # Best-effort cleanup, then propagate as a clean error
        for f in [raw_video, synth_wav, padded_wav]:
            if os.path.exists(f):
                try: os.remove(f)
                except Exception: pass
        raise RuntimeError(f"ffmpeg mux timed out after {FFMPEG_TIMEOUT}s")

    if r3.returncode != 0:
        for f in [raw_video, synth_wav, padded_wav]:
            if os.path.exists(f):
                try: os.remove(f)
                except Exception: pass
        raise RuntimeError(f"ffmpeg mux failed:\n{r3.stderr}")

    log.info(f"[video] final video written to {final_path}")

    # Cleanup temp files
    for f in [raw_video, synth_wav, padded_wav]:
        if os.path.exists(f):
            try: os.remove(f)
            except Exception as e:
                log.warning(f"Cleanup failed for {f}: {e}")

    return final_path
