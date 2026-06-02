"""
debug_musan_noise.py

Apply REALISTIC noise using MUSAN to a clean WAV file.

Usage:
python noise_debug.py --input /export/clusterdata/mflara/maestro/2014/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--1.wav --musan_dir /export/clusterdata/mflara/musan/musan --output_wav noisy.wav  --output_mel noisy_mel.npy --noisy_versions 3
"""
"""
2
"""
"""
debug_precompute.py

Test the full precompute augmentation pipeline on a single WAV file.
Saves both the noisy WAV (for listening) and the mel spectrogram (for inspection).

Usage:
python noise_debug.py \
  --input /path/to/piano.wav \
  --musan_dir /path/to/musan \
  --output_wav noisy.wav \
  --output_mel noisy_mel.npy \
  --noisy_versions 3
"""

import argparse
import os
import glob
import random
import subprocess
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SAMPLE_RATE = 16000
HOP_LENGTH  = 128
N_FFT       = 1024
N_MELS      = 512


# ─────────────────────────────────────────────
# Load audio
# ─────────────────────────────────────────────

def load_audio(path):
    wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    print(f"[LOAD] {os.path.basename(path)} | sr={sr} | duration={len(wav)/SAMPLE_RATE:.2f}s | samples={len(wav)}")
    return wav


# ─────────────────────────────────────────────
# Load MUSAN
# ─────────────────────────────────────────────

def load_musan(musan_dir):
    noise_files  = glob.glob(os.path.join(musan_dir, "noise",  "**", "*.wav"), recursive=True)
    speech_files = glob.glob(os.path.join(musan_dir, "speech", "**", "*.wav"), recursive=True)
    print(f"[MUSAN] noise: {len(noise_files)} files | speech: {len(speech_files)} files")
    return noise_files, speech_files


# ─────────────────────────────────────────────
# SNR mixing
# ─────────────────────────────────────────────

def mix_snr(clean, noise, snr_db):
    clean_rms = np.sqrt(np.mean(clean ** 2)) + 1e-8
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
    scale = clean_rms / (noise_rms * (10 ** (snr_db / 20)))
    return clean + scale * noise


# ─────────────────────────────────────────────
# Piecewise background noise
# ─────────────────────────────────────────────

def add_background_piecewise(wav, files, sample_rate, snr_range, apply_prob=0.7, label="noise"):
    if not files:
        return wav

    wav = wav.copy()
    segment_len = int(2.0 * sample_rate)
    fade_len    = int(0.02 * sample_rate)
    fade_in     = np.linspace(0, 1, fade_len)
    fade_out    = np.linspace(1, 0, fade_len)

    i      = 0
    seg_id = 0

    while i < len(wav):
        if random.random() < apply_prob:
            path  = random.choice(files)
            noise, _ = librosa.load(path, sr=sample_rate, mono=True)

            if len(noise) < segment_len:
                repeats = int(np.ceil(segment_len / len(noise)))
                noise   = np.tile(noise, repeats)

            n_start   = random.randint(0, len(noise) - segment_len)
            noise_seg = noise[n_start : n_start + segment_len].copy()
            noise_seg[:fade_len]  *= fade_in
            noise_seg[-fade_len:] *= fade_out

            end       = min(i + segment_len, len(wav))
            audio_seg = wav[i:end]
            snr       = random.uniform(*snr_range)

            print(f"  [{label}] seg {seg_id:02d} | {os.path.basename(path):<40} | SNR={snr:.1f} dB")

            wav[i:end] = mix_snr(audio_seg, noise_seg[: len(audio_seg)], snr)
        else:
            print(f"  [{label}] seg {seg_id:02d} | skipped (apply_prob)")

        i      += segment_len
        seg_id += 1

    return wav


# ─────────────────────────────────────────────
# Phone EQ
# ─────────────────────────────────────────────

def phone_eq(wav, sample_rate):
    sos = butter(4, 100, btype="high", fs=sample_rate, output="sos")
    out = sosfilt(sos, wav).astype(np.float32)
    print(f"  [EQ] high-pass 100 Hz applied")
    return out


# ─────────────────────────────────────────────
# Room reverb
# ─────────────────────────────────────────────

def add_reverb(wav, sample_rate):
    try:
        import pyroomacoustics as pra

        rt60     = random.uniform(0.2, 0.8)
        room_dim = [random.uniform(3, 8), random.uniform(3, 8), random.uniform(2.5, 4)]
        src_pos  = [random.uniform(1, d - 1) for d in room_dim]
        mic_pos  = [random.uniform(1, d - 1) for d in room_dim]

        print(f"  [REVERB] rt60={rt60:.2f}s | room={[round(d,1) for d in room_dim]} | src={[round(p,1) for p in src_pos]} | mic={[round(p,1) for p in mic_pos]}")

        room = pra.ShoeBox(room_dim, fs=sample_rate, materials=pra.Material(0.2), max_order=12)
        room.add_source(src_pos, signal=wav)
        room.add_microphone(np.array(mic_pos).reshape(3, 1))
        room.simulate()

        reverbed = room.mic_array.signals[0][: len(wav)]
        return reverbed.astype(np.float32)

    except ImportError:
        print("  [REVERB] pyroomacoustics not installed — skipped")
        return wav


# ─────────────────────────────────────────────
# Codec compression
# ─────────────────────────────────────────────

def codec_compress(wav, sample_rate, bitrate="64k"):
    try:
        cmd_enc = [
            "ffmpeg", "-y",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "aac", "-b:a", bitrate,
            "-f", "adts", "pipe:1",
        ]
        cmd_dec = [
            "ffmpeg",
            "-f", "aac", "-i", "pipe:0",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
            "pipe:1",
        ]
        enc     = subprocess.run(cmd_enc, input=wav.tobytes(), capture_output=True)
        dec     = subprocess.run(cmd_dec, input=enc.stdout,   capture_output=True)
        decoded = np.frombuffer(dec.stdout, dtype=np.float32)

        if len(decoded) == 0:
            print("  [CODEC] ffmpeg returned empty audio — skipped")
            return wav

        if len(decoded) >= len(wav):
            decoded = decoded[: len(wav)]
        else:
            decoded = np.concatenate([decoded, np.zeros(len(wav) - len(decoded), dtype=np.float32)])

        print(f"  [CODEC] AAC {bitrate} roundtrip applied")
        return decoded

    except Exception as e:
        print(f"  [CODEC] failed ({e}) — skipped")
        return wav


# ─────────────────────────────────────────────
# Full augmentation pipeline
# ─────────────────────────────────────────────

def augment(wav, noise_files, speech_files, sample_rate, version_id=0):
    print(f"\n── version {version_id} ───────────────────────────────────────")

    rms = np.sqrt(np.mean(wav ** 2)) + 1e-8
    wav = (wav / rms * 0.1).astype(np.float32)
    print(f"  [NORM] RMS normalised: target RMS=0.1")

    if random.random() < 0.8:
        wav = add_reverb(wav, sample_rate)
    else:
        print("  [REVERB] skipped (prob)")

    if random.random() < 0.6:
        wav = phone_eq(wav, sample_rate)
    else:
        print("  [EQ] skipped (prob)")

    if noise_files and random.random() < 0.7:
        print(f"  [NOISE] applying piecewise background noise...")
        wav = add_background_piecewise(wav, noise_files, sample_rate, snr_range=(15, 35), label="noise")
    else:
        print("  [NOISE] skipped (prob or no files)")

    if speech_files and random.random() < 0.3:
        print(f"  [SPEECH] applying piecewise speech babble...")
        wav = add_background_piecewise(wav, speech_files, sample_rate, snr_range=(20, 35), label="speech")
    else:
        print("  [SPEECH] skipped (prob or no files)")

    if random.random() < 0.2:
        wav = codec_compress(wav, sample_rate)
    else:
        print("  [CODEC] skipped (prob)")

    print(f"  [DONE] output RMS={np.sqrt(np.mean(wav**2)):.4f} | peak={np.max(np.abs(wav)):.4f}")
    return wav


# ─────────────────────────────────────────────
# Mel computation
# ─────────────────────────────────────────────

def to_mel(wav):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        n_fft=N_FFT, n_mels=N_MELS, power=2.0,
    )
    mel = np.log(mel + 1e-6)
    return mel.T.astype(np.float16)   # (T, N_MELS)




def main(args):
    wav = load_audio(args.input)
    noise_files, speech_files = load_musan(args.musan_dir)

    wav_stem = os.path.splitext(args.output_wav)[0]
    mel_stem = os.path.splitext(args.output_mel)[0]  # ← derive mel stem from --output_mel

    for v in range(args.noisy_versions):
        noisy_wav = augment(wav, noise_files, speech_files, SAMPLE_RATE, version_id=v)

        wav_path = f"{wav_stem}_v{v}.wav"
        mel_path = f"{mel_stem}_v{v}.npy"  # ← use mel_stem

        sf.write(wav_path, noisy_wav, SAMPLE_RATE)
        print(f"  [SAVE] wav → {wav_path}")

        mel = to_mel(noisy_wav)
        np.save(mel_path, mel)
        print(f"  [SAVE] mel → {mel_path}  shape={mel.shape}  dtype={mel.dtype}")

    print("\nAll versions done.")


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           type=str, required=True)
    parser.add_argument("--musan_dir",       type=str, required=True)
    parser.add_argument("--output_wav",      type=str, default="noisy.wav")
    parser.add_argument("--output_mel",      type=str, default="noisy_mel.npy")  # ← add this
    parser.add_argument("--noisy_versions",  type=int, default=3)
    args = parser.parse_args()
    main(args)