"""
Used to create a deterministic noisy test set from MAESTRO.
Outputs WAV files, so mels need to be recomputed later.
"""

# Loading the libraries

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, iirpeak, lfilter
import pyroomacoustics as pra



# Seeding

def set_global_seed(seed: int):
    # Use the same seed for all random operations to ensure full determinism.
    # Used seed -> 42
    random.seed(seed)
    np.random.seed(seed)



# MUSAN file discovery

def load_musan_files(musan_dir: str):
    musan_path   = Path(musan_dir)
    noise_files  = sorted(musan_path.rglob("noise/**/*.wav"))
    speech_files = sorted(musan_path.rglob("speech/**/*.wav"))

    if len(noise_files) == 0:
        raise FileNotFoundError(f"No noise WAV files found under {musan_dir}/noise/")
    if len(speech_files) == 0:
        raise FileNotFoundError(f"No speech WAV files found under {musan_dir}/speech/")

    print(f"MUSAN  noise: {len(noise_files)} files")
    print(f"MUSAN speech: {len(speech_files)} files")
    return noise_files, speech_files



# Audio helpers


def load_audio(path, sr: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def normalize_rms(wav: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-8
    return wav / rms * target_rms


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # Mixes clean + noise at a given SNR (dB). Both must be the same length

    clean_rms = np.sqrt(np.mean(clean ** 2)) + 1e-8
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
    scale = clean_rms / (noise_rms * (10 ** (snr_db / 20.0)))
    return clean + scale * noise


def get_noise_segment(
    noise_files: list,
    length: int,
    sr: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    
    # Picks a random noise file and extract a segment of exactly 'length' samples
    # Uses the per-song RandomState so selection is deterministic per (song, condition)

    path  = noise_files[rng.randint(0, len(noise_files))]
    noise = load_audio(path, sr)
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    max_start = len(noise) - length
    start     = rng.randint(0, max_start + 1) if max_start > 0 else 0
    return noise[start : start + length]



# Degradation functions

def apply_phone_eq(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Realistic phone mic coloration: slightly OOD relative to training
    (which used a plain 100 Hz high-pass only):
      - High-pass at 300 Hz  (stricter than training's 100 Hz)
      - Presence boost +3 dB at 3 kHz  (mic coloration not seen in training)
      - Low-pass at 8 kHz  (phone mics roll off at high frequencies)
    """
    # High-pass at 300 Hz
    sos_hp = butter(4, 300, btype="high", fs=sr, output="sos")
    wav = sosfilt(sos_hp, wav).astype(np.float32)
 
    # Presence boost at 3 kHz
    b, a = iirpeak(3000, Q=1.5, fs=sr)
    wav = lfilter(b, a, wav).astype(np.float32)
 
    # Low-pass at 7500 Hz (must be strictly below Nyquist = sr/2)
    sos_lp = butter(2, 7500, btype="low", fs=sr, output="sos")
    wav = sosfilt(sos_lp, wav).astype(np.float32)
 
    return wav


def _pra_reverb(
    wav: np.ndarray,
    sr: int,
    rng: np.random.RandomState,
    room_dim_range: tuple,
    absorption: float,
) -> np.ndarray:
    """
    Reverb conditions use pyroomacoustics (same library as the training augmentation)
    so the test domain matches the training domain.  Two reverb conditions are
    included:
        - reverb_small : small domestic room (within training distribution)
        - reverb_large : large room, less absorptive (harder than training)
    """
    try:

        lo, hi = room_dim_range
        room_dim = [
            float(rng.uniform(lo[0], hi[0])),
            float(rng.uniform(lo[1], hi[1])),
            float(rng.uniform(lo[2], hi[2])),
        ]
        src_pos = [float(rng.uniform(0.5, d - 0.5)) for d in room_dim]
        mic_pos = [float(rng.uniform(0.5, d - 0.5)) for d in room_dim]

        room = pra.ShoeBox(
            room_dim,
            fs=sr,
            materials=pra.Material(absorption),
            max_order=12,
        )
        room.add_source(src_pos, signal=wav)
        room.add_microphone(np.array(mic_pos).reshape(3, 1))
        room.simulate()

        out = room.mic_array.signals[0][: len(wav)].astype(np.float32)

        # Preserve input loudness so SNR mixing further down the chain is
        # not distorted by reverb-induced level changes
        rms_in  = np.sqrt(np.mean(wav ** 2)) + 1e-8
        rms_out = np.sqrt(np.mean(out ** 2)) + 1e-8
        return out * (rms_in / rms_out)

    except ImportError:
        print(" Pyroomacoustics not installed. Fallback: using normalised exponential IR")
        decay_len = int(0.15 * sr) if absorption >= 0.3 else int(0.35 * sr)
        ir  = np.exp(-np.linspace(0, 6, decay_len)).astype(np.float32)
        ir  = ir / ir.sum()   # normalise: preserves loudness
        out = np.convolve(wav, ir, mode="full")[: len(wav)]
        rms_in  = np.sqrt(np.mean(wav ** 2)) + 1e-8
        rms_out = np.sqrt(np.mean(out ** 2)) + 1e-8
        return (out * (rms_in / rms_out)).astype(np.float32)


def apply_reverb_small(wav: np.ndarray, sr: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Small domestic room: 3–5 m sides, 2.5–3.5 m ceiling
    Absorption 0.35: moderately absorptive (carpet, furniture)
    Within training distribution (training: 3–8 m rooms, absorption 0.2)
    """
    return _pra_reverb(
        wav, sr, rng,
        room_dim_range=([3, 3, 2.5], [5, 5, 3.5]),
        absorption=0.35,
    )


def apply_reverb_large(wav: np.ndarray, sr: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Large room: 8–12 m sides, 3–5 m ceiling
    Absorption 0.15:reflective surfaces (hard floors, bare walls)
    Harder than training, tests generalisation to unseen room sizes
    """
    return _pra_reverb(
        wav, sr, rng,
        room_dim_range=([8, 8, 3], [12, 12, 5]),
        absorption=0.15,
    )


# Condition list

CONDITIONS = [
    "clean",

    # Additive noise: training used SNR 15–35 dB
    "noise_20db",    # within training range: mild challenge
    "noise_10db",    # below training range: moderate challenge
    "noise_5db",     # well below training range: severe, shows degradation curve

    # Speech interference at 15 dB: within training range
    "speech_15db",

    # Phone mic coloration: slightly OOD (300 Hz HP vs training 100 Hz HP)
    "phone_eq",

    # Reverb: pyroomacoustics, same engine as training
    "reverb_small",    # within training distribution
    "reverb_large",    # harder than training, tests generalisation

    # Realistic phone recording chain
    "phone_simulation",
]



# Condition dispatcher

def apply_condition(
    wav: np.ndarray,
    sr: int,
    condition: str,
    noise_files: list,
    speech_files: list,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Apply one degradation condition to a normalised waveform.
    All randomness goes through 'rng' so results are deterministic
    given the same seed.
    """
    wav = normalize_rms(wav)

    if condition == "clean":
        return wav

    if condition == "noise_20db":
        noise = get_noise_segment(noise_files, len(wav), sr, rng)
        return mix_at_snr(wav, noise, snr_db=20)

    if condition == "noise_10db":
        noise = get_noise_segment(noise_files, len(wav), sr, rng)
        return mix_at_snr(wav, noise, snr_db=10)

    if condition == "noise_5db":
        noise = get_noise_segment(noise_files, len(wav), sr, rng)
        return mix_at_snr(wav, noise, snr_db=5)

    if condition == "speech_15db":
        noise = get_noise_segment(speech_files, len(wav), sr, rng)
        return mix_at_snr(wav, noise, snr_db=15)

    if condition == "phone_eq":
        return apply_phone_eq(wav, sr)

    if condition == "reverb_small":
        return apply_reverb_small(wav, sr, rng)

    if condition == "reverb_large":
        return apply_reverb_large(wav, sr, rng)

    if condition == "phone_simulation":
        # Realistic chain matching actual phone recording conditions:
        #   1. phone EQ (slightly OOD: 300 Hz HP + presence boost)
        #   2. small room reverb (pyroomacoustics, within training dist)
        #   3. additive noise at 15 dB (hard end of training range, not below it)

        wav = apply_phone_eq(wav, sr)
        wav = apply_reverb_small(wav, sr, rng)
        noise = get_noise_segment(noise_files, len(wav), sr, rng)
        wav = mix_at_snr(wav, noise, snr_db=15)
        return wav

    raise ValueError(f"Unknown condition: '{condition}'")



# Main

def main(args):

    set_global_seed(args.seed)

    maestro_root = Path(args.maestro_root)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = maestro_root / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO CSV not found at {csv_path}")

    df      = pd.read_csv(csv_path)
    df_test = df[df["split"] == "test"].copy()

    if len(df_test) < args.num_songs:
        raise ValueError(
            f"Requested {args.num_songs} songs but MAESTRO test split "
            f"only has {len(df_test)}"
        )

    df_selected = df_test.sample(
        n=args.num_songs, random_state=args.seed
    ).reset_index(drop=True)

    print(f"Selected {len(df_selected)} songs from MAESTRO test split")

    # Manifest 
    manifest_cols = ["audio_filename", "midi_filename", "composer", "title", "year"]
    manifest_cols = [c for c in manifest_cols if c in df_selected.columns]
    df_selected[manifest_cols].to_csv(output_dir / "manifest.csv", index=False)
    print(f"Manifest saved → {output_dir / 'manifest.csv'}")

    # Build config
    build_config = {
        "seed":                              args.seed,
        "num_songs":                         args.num_songs,
        "sample_rate":                       args.sample_rate,
        "conditions":                        CONDITIONS,
        "maestro_root":                      str(maestro_root),
        "maestro_split":                     "test",
        "musan_dir":                         str(args.musan_dir),
        "reverb_engine":                     "pyroomacoustics (fallback: normalised exponential IR)",
        "reverb_small_absorption":           0.35,
        "reverb_large_absorption":           0.15,
        "phone_eq_highpass_hz":              300,
        "phone_eq_presence_boost_hz":        3000,
        "phone_simulation_snr_db":           20,
    }
    with open(output_dir / "build_config.json", "w") as fh:
        json.dump(build_config, fh, indent=2)
    print(f"Build config saved → {output_dir / 'build_config.json'}")

    # Output directories 
    for cond in CONDITIONS:
        (output_dir / cond).mkdir(exist_ok=True)

    # MUSAN 
    noise_files, speech_files = load_musan_files(args.musan_dir)

    # Process
    print(f"\nProcessing {len(df_selected)} songs × {len(CONDITIONS)} conditions …\n")

    for row_idx, (_, row) in enumerate(df_selected.iterrows()):

        audio_path = maestro_root / row["audio_filename"]
        stem       = Path(row["audio_filename"]).stem

        if not audio_path.exists():
            print(f" Audio file not found, skipping: {audio_path}")
            continue

        wav = load_audio(audio_path, args.sample_rate)
        print(f"  [{row_idx + 1:2d}/{len(df_selected)}] {stem}")

        for cond_idx, cond in enumerate(CONDITIONS):

            # Independent RNG per (song, condition):
            #   seed + row_idx * 1000 + cond_idx
            # Adding or removing a condition never changes any other combination.
            rng = np.random.RandomState(args.seed + row_idx * 1000 + cond_idx)

            out = apply_condition(
                wav, args.sample_rate, cond,
                noise_files, speech_files,
                rng,
            )

            out_path = output_dir / cond / f"{stem}.wav"
            sf.write(str(out_path), out, args.sample_rate)

        print(f"DONE: All conditions written")

    print(f"\nTest set complete: {output_dir}")
    print(f" Conditions : {CONDITIONS}")
    print(f" Songs: {len(df_selected)}")
    print(f" Total files: {len(df_selected) * len(CONDITIONS)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a deterministic noisy test set from MAESTRO + MUSAN."
    )
    parser.add_argument("--maestro_root", type=str, required=True,
                        help="Root of MAESTRO dataset (contains maestro-v3.0.0.csv)")
    parser.add_argument("--musan_dir",    type=str, required=True,
                        help="Root of MUSAN dataset (contains noise/ and speech/)")
    parser.add_argument("--output_dir",   type=str, required=True,
                        help="Where to write the test set WAV files")
    parser.add_argument("--num_songs",    type=int, default=20)
    parser.add_argument("--sample_rate",  type=int, default=16000)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    main(args)