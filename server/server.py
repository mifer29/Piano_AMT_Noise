import os
import torch
import tempfile
import base64
import sys
import shutil
import asyncio
import logging
import uuid
import numpy as np
import pretty_midi

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_robust_model.model import AudioTransformer, greedy_decode

from config import (
    CHECKPOINT_PATH, SOUNDFONT_PATH, DEVICE,
    N_MELS, MAX_INPUT_FRAMES, MAX_OUTPUT_TOKENS, STRIDE, HOP_LENGTH, SAMPLE_RATE,
    MAX_UPLOAD_BYTES, MAX_AUDIO_SECONDS,
)
from vocab import vocab, bos_id, eos_id
from audio import compute_mel, get_audio_duration, tokens_to_notes, deduplicate_notes
from midi_utils import notes_to_midi, safe_estimate_tempo
from video import generate_video


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcribe")


# Startup checks: fail fast and loudly

def check_startup_requirements():
    """Verify all external dependencies are present before serving requests.
    Logs clearly and exits if anything critical is missing."""
    problems = []

    # Required binaries on PATH
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            problems.append(f"  ✗ '{binary}' not found on PATH")
        else:
            log.info(f"  {binary} found at {shutil.which(binary)}")

    # fluidsynth is required for audio; we degrade gracefully if missing,
    # but warn loudly so the defense isn't surprised by silent videos.
    if shutil.which("fluidsynth") is None:
        log.warning(" 'fluidsynth' not found on PATH — videos will be silent")
    else:
        log.info(f" fluidsynth found at {shutil.which('fluidsynth')}")

    # Required files
    if not os.path.exists(CHECKPOINT_PATH):
        problems.append(f" checkpoint not found: {CHECKPOINT_PATH}")
    else:
        log.info(f" checkpoint found: {CHECKPOINT_PATH}")

    if not os.path.exists(SOUNDFONT_PATH):
        log.warning(f"  soundfont not found: {SOUNDFONT_PATH} — videos will be silent")
    else:
        log.info(f" soundfont found: {SOUNDFONT_PATH}")

    if problems:
        log.error("Startup checks FAILED:")
        for p in problems:
            log.error(p)
        log.error("Server cannot start. Fix the issues above and retry.")
        sys.exit(1)

    log.info(f"Device: {DEVICE}")
    log.info("Startup checks passed.")

log.info("=" * 60)
log.info("Piano Transcriber Server starting…")
log.info("=" * 60)
check_startup_requirements()


# Load model

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = AudioTransformer(n_mels=N_MELS, vocab_size=len(vocab))
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    log.info(f"Model loaded from {CHECKPOINT_PATH}")
except Exception as e:
    log.error(f"Failed to load model: {e}")
    sys.exit(1)




# Only one transcription at a time: we have one model on one device, and
# parallel requests would either deadlock the CPU or produce garbled outputs.
inference_lock = asyncio.Lock()


# Transcription

@torch.no_grad()
def transcribe_audio(audio_path):
    """
    Sliding-window inference with 25% overlap. Every window emits its decoded
    notes into a global list, and a post-hoc deduplication step then merges the
    same-pitch boundary duplicates introduced by the overlap (50 ms tolerance,
    keeping the longer note). Matches evaluate.py:transcribe_song.
    """
    mel     = compute_mel(audio_path)
    mel_len = mel.shape[0]

    seg_dur     = MAX_INPUT_FRAMES * HOP_LENGTH / SAMPLE_RATE          # 4.096 s
    stride_time = STRIDE           * HOP_LENGTH / SAMPLE_RATE          # 3.072 s

    all_notes   = []
    frame_pos   = 0
    time_offset = 0.0
    seg_idx     = 0

    while frame_pos < mel_len:

        seg = mel[frame_pos : frame_pos + MAX_INPUT_FRAMES]
        if seg.shape[0] < MAX_INPUT_FRAMES:
            seg = np.pad(seg, ((0, MAX_INPUT_FRAMES - seg.shape[0]), (0, 0)))

        audio_t  = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        token_ids = greedy_decode(
            model, audio_t,
            bos_id=bos_id, eos_id=eos_id,
            max_len=MAX_OUTPUT_TOKENS
        )

        seg_notes = tokens_to_notes(token_ids, segment_duration=seg_dur)

        for onset, offset, pitch, velocity in seg_notes:

            # Skip invalid / out-of-bounds notes
            if offset <= 0:
                continue
            if onset > seg_dur + 0.05:
                continue

            onset_global  = onset  + time_offset
            offset_global = offset + time_offset

            if offset_global <= onset_global:
                offset_global = onset_global + 0.01

            if offset_global - onset_global < 0.01:
                continue

            all_notes.append((onset_global, offset_global, pitch, velocity))

        frame_pos   += STRIDE
        time_offset += stride_time
        seg_idx     += 1

    # Sort globally, then merge overlap-induced duplicates
    all_notes = sorted(all_notes, key=lambda x: x[0])
    all_notes = deduplicate_notes(all_notes)
    return all_notes



# Helper: streamed bounded upload

async def save_upload_bounded(upload: UploadFile, dest_path: str, max_bytes: int) -> int:
    """
    Stream the upload to disk in chunks, aborting if it exceeds max_bytes.
    Returns the number of bytes written.
    Raises HTTPException(413) if the upload is too large.
    """
    written = 0
    chunk_size = 64 * 1024
    with open(dest_path, "wb") as out:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                out.close()
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (max {max_bytes // (1024*1024)} MB)"
                )
            out.write(chunk)
    return written



# API

app = FastAPI()

@app.get("/health")
async def health():
    """Lightweight liveness probe, useful for the app's online indicator."""
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    req_id = uuid.uuid4().hex[:8]
    log.info(f"[{req_id}] /transcribe received: filename={file.filename!r}")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    # Save upload to a temp file with bounded size
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_path = tmp.name

    midi_path = audio_path + ".mid"
    video_path = audio_path + ".mp4"

    try:
        # Size check (streaming, bounded)
        try:
            written = await save_upload_bounded(file, audio_path, MAX_UPLOAD_BYTES)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"[{req_id}] Failed to save upload: {e}")
            return JSONResponse({"error": "Failed to save upload"}, status_code=500)

        log.info(f"[{req_id}] Saved {written} bytes")

        if written < 1024:
            return JSONResponse({"error": "Audio file too small"}, status_code=400)

        # Duration check
        duration = get_audio_duration(audio_path)
        log.info(f"[{req_id}] Probed duration: {duration:.2f}s")
        if duration > MAX_AUDIO_SECONDS:
            log.warning(f"[{req_id}] Rejected: duration {duration:.1f}s > {MAX_AUDIO_SECONDS}s")
            return JSONResponse(
                {"error": f"Audio too long ({duration:.1f}s). Max {MAX_AUDIO_SECONDS}s."},
                status_code=413
            )
        if duration > 0:
            log.info(f"[{req_id}] Duration: {duration:.2f}s")

        # Inference + video (serialized: one CPU)
        async with inference_lock:
            log.info(f"[{req_id}] Acquired inference lock, starting transcription")

            # 1. Transcribe
            notes = transcribe_audio(audio_path)
            if not notes:
                log.info(f"[{req_id}] No notes detected")
                return JSONResponse({"error": "No notes detected"}, status_code=422)
            log.info(f"[{req_id}] Transcribed {len(notes)} notes")

            # 2. MIDI
            pm  = notes_to_midi(notes, midi_path)
            bpm = safe_estimate_tempo(pm)
            log.info(f"[{req_id}] Estimated tempo: {bpm:.1f} BPM")

            # 3. Video
            pm_loaded = pretty_midi.PrettyMIDI(midi_path)
            pm_notes  = [n for inst in pm_loaded.instruments for n in inst.notes]
            try:
                final_video = generate_video(pm_notes, midi_path, video_path)
            except ValueError as e:
                # Raised if ghost-note filtering removes everything
                log.warning(f"[{req_id}] Video generation skipped: {e}")
                return JSONResponse(
                    {"error": "Detected notes were all filtered as artifacts"},
                    status_code=422
                )
            except RuntimeError as e:
                log.error(f"[{req_id}] Video generation failed: {e}")
                return JSONResponse(
                    {"error": "Video generation failed"},
                    status_code=500
                )

        # 4. Encode
        try:
            with open(midi_path,   "rb") as f: midi_b64  = base64.b64encode(f.read()).decode()
            with open(final_video, "rb") as f: video_b64 = base64.b64encode(f.read()).decode()
        except IOError as e:
            log.error(f"[{req_id}] Failed to read output files: {e}")
            return JSONResponse({"error": "Failed to read outputs"}, status_code=500)

        log.info(f"[{req_id}] Success: returning response")
        return JSONResponse({"bpm": bpm, "midi": midi_b64, "video": video_b64})

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"[{req_id}] Unexpected error")
        return JSONResponse({"error": f"Server error: {e}"}, status_code=500)

    finally:
        for path in [
            audio_path,
            midi_path,
            audio_path + "_raw.mp4",
            video_path,
            audio_path + "_synth.wav",
            audio_path + "_padded.wav",
        ]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    log.warning(f"[{req_id}] Cleanup failed for {path}: {e}")
