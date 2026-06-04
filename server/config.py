import os
import torch
from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_PATH = os.getenv("model_path")
SOUNDFONT_PATH  = os.getenv("soundfont_path")

SAMPLE_RATE       = 16000
HOP_LENGTH        = 128
N_MELS            = 512
MAX_INPUT_FRAMES  = 512
MAX_OUTPUT_TOKENS = 512

# 25% overlap, matching the evaluation pipeline and the thesis (Section 5.3.3).
# Notes are emitted from every window and a post-hoc deduplication step then
# merges the boundary duplicates introduced by the overlap, exactly as in
# evaluate.py. This keeps the deployed pipeline identical to the one that
# produced the reported metrics.
OVERLAP_RATIO     = 0.25
STRIDE            = max(1, int(MAX_INPUT_FRAMES * (1 - OVERLAP_RATIO)))   # = 384

# Request limits 
MAX_UPLOAD_BYTES   = 300 * 1024 * 1024   # 300 MB
MAX_AUDIO_SECONDS  = 300.0               # reject anything longer

# Subprocess timeouts (seconds)
FLUIDSYNTH_TIMEOUT = 300
FFMPEG_TIMEOUT     = 300
FFPROBE_TIMEOUT    = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Video config
FPS        = 30
WIDTH      = 1280
HEIGHT     = 720
KEYBOARD_H = 160
SPEED      = 150      # pixels per second
LEAD_IN    = 2.0
LEAD_OUT   = 2.0
SPLIT_PITCH= 60       # >= cyan (right hand), < amber (left hand)

MIDI_MIN = 21
MIDI_MAX = 108
WHITE_OFFSETS = [0, 2, 4, 5, 7, 9, 11]
BLACK_OFFSETS = [1, 3, 6, 8, 10]

NUM_WHITE_KEYS = sum(
    1 for m in range(MIDI_MIN, MIDI_MAX + 1)
    if (m % 12) in WHITE_OFFSETS
)

# Colours (BGR)
BG_COLOR          = (18,  18,  18)
WHITE_KEY_COLOR   = (240, 240, 235)
BLACK_KEY_COLOR   = (30,  30,  30)
NOTE_COLOR_RIGHT  = (100, 210, 255)   # cyan
NOTE_COLOR_LEFT   = (255, 160,  80)   # amber
NOTE_BORDER       = (255, 255, 255)
HIT_LINE_COLOR    = (255, 255, 255)
DIVIDER_COLOR     = (60,  60,  60)
