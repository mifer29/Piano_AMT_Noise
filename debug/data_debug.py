'''
Sanity-checks one dataset sample: validates the target token sequence structure, the
alignment between mel segment duration and event time bins, and the model's loss/logits at init.
'''
import math
import numpy as np
import torch
import torch.nn as nn
import random
from pathlib import Path
from maestro_dataset import MAESTROSeq2SeqDataset, collate_fn
from model import AudioTransformer, shift_tokens_right
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
load_dotenv()


# CONFIG


maestro_root = os.getenv("maestro_root")
events_dir   = os.getenv("events_dir")
mel_dir      = os.getenv("mel_dir")

TEST_TOKENS     = True
TEST_ALIGNMENT  = True
TEST_MODEL_INIT = True

random.seed(42)


# Create dataset + grab one sample


dataset = MAESTROSeq2SeqDataset(
    maestro_path=maestro_root,
    events_dir=events_dir,
    mel_dir=mel_dir,
    split="train"
)

print(f"Dataset size: {len(dataset)}")
print(f"Vocab size:   {len(dataset.vocab)}")
print(f"PAD id: {dataset.pad_id} | BOS id: {dataset.bos_id} | EOS id: {dataset.eos_id}")

random.seed(42)
sample = dataset[0]

audio_features = sample["audio_features"]
target_ids     = sample["target_ids"]



# TEST 1: Token sequence check


def test_tokens():
    print("\n" + "=" * 50)
    print("TEST 1: TOKEN SEQUENCE")
    print("=" * 50)

    tokens = target_ids.tolist()

    print(f"Sequence length:   {len(tokens)}")
    print(f"EOS present:       {dataset.eos_id in tokens}")
    print(f"BOS present:       {dataset.bos_id in tokens}")
    print(f"PAD in middle:     {dataset.pad_id in tokens[:-1]}")

    type_counts = {}
    for t in tokens:
        token_type = dataset.vocab[t].split("_")[0]
        type_counts[token_type] = type_counts.get(token_type, 0) + 1
    print(f"Token type counts: {type_counts}")

    print("\nFirst 60 tokens decoded:")
    for i, t in enumerate(tokens[:60]):
        print(f"  [{i:>3}] {dataset.vocab[t]}")

    print("\nChecking time->velocity->note structure...")
    errors = 0
    for i in range(len(tokens) - 1):
        tok = dataset.vocab[tokens[i]]
        if tok.startswith("time_"):
            nxt = dataset.vocab[tokens[i + 1]]
            if not nxt.startswith("velocity_"):
                print(f"  WARNING at pos {i}: time not followed by velocity. Got: {nxt}")
                errors += 1
    if errors == 0:
        print("  OK Structure looks correct.")
    else:
        print(f"  FAIL {errors} structural issues found.")



# TEST 2: Mel-event alignment check


def test_alignment():
    print("\n" + "=" * 50)
    print("TEST 2: MEL-EVENT ALIGNMENT")
    print("=" * 50)

    tokens = target_ids.tolist()
    time_tokens = [dataset.vocab[t] for t in tokens if dataset.vocab[t].startswith("time_")]

    if not time_tokens:
        print("FAIL No time tokens found. Cannot check alignment.")
        return

    first_bin = int(time_tokens[0].split("_")[1])
    last_bin  = int(time_tokens[-1].split("_")[1])
    max_bin   = max(int(t.split("_")[1]) for t in time_tokens)

    first_sec = first_bin * dataset.time_resolution
    last_sec  = last_bin  * dataset.time_resolution
    segment_duration = dataset.max_input_frames * dataset.hop_length / dataset.sample_rate

    print(f"Segment duration:      {segment_duration:.3f}s")
    print(f"First event time:      {first_sec:.3f}s  (bin {first_bin})")
    print(f"Last event time:       {last_sec:.3f}s  (bin {last_bin})")
    print(f"Max time bin used:     {max_bin}  (limit: {dataset.max_time_bins - 1})")

    if first_bin < 0:
        print("FAIL Negative time bin found.")
    else:
        print("OK  All time bins non-negative.")

    if max_bin >= dataset.max_time_bins:
        print("FAIL Time bin exceeds vocabulary limit.")
    else:
        print("OK  All time bins within vocabulary range.")

    if last_sec > segment_duration + dataset.time_resolution:
        print(f"FAIL Events extend beyond mel segment by {last_sec - segment_duration:.3f}s")
    else:
        print("OK  Events contained within mel segment.")



# TEST 3: Model output distribution at init


def test_model_init():
    print("\n" + "=" * 50)
    print("TEST 3: MODEL INIT DISTRIBUTION")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    batch = next(iter(loader))

    audio   = batch["audio_features"].to(device)
    targets = batch["target_ids"].to(device)

    model = AudioTransformer(
        n_mels=512,
        vocab_size=len(dataset.vocab)
    ).to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)

    with torch.no_grad():
        decoder_input = shift_tokens_right(targets, dataset.bos_id)
        logits = model(audio, decoder_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

    expected_init_loss = math.log(len(dataset.vocab))

    print(f"Expected loss at init (ln(vocab)): {expected_init_loss:.4f}")
    print(f"Actual loss at init:               {loss.item():.4f}")
    print(f"Ratio actual/expected:             {loss.item() / expected_init_loss:.3f}")

    print(f"\nLogit mean: {logits.mean().item():.4f}")
    print(f"Logit std:  {logits.std().item():.4f}")

    print(f"\nDecoder input[0][:10]: {decoder_input[0][:10].tolist()}")
    print(f"Target[0][:10]:        {targets[0][:10].tolist()}")
    print(f"decoder_input[0][0] == bos_id: {decoder_input[0][0].item() == dataset.bos_id}")
    print(f"decoder_input[0][1:6] == targets[0][:5]: {decoder_input[0][1:6].tolist() == targets[0][:5].tolist()}")

    if abs(loss.item() - expected_init_loss) < 1.0:
        print("\nOK  Model init looks healthy.")
    elif loss.item() > expected_init_loss * 2:
        print("\nFAIL Loss is much higher than expected. Likely a mask or init bug.")
    else:
        print("\nWARN Loss is somewhat above expected. Monitor closely.")


# RUN TESTS


if TEST_TOKENS:
    test_tokens()

if TEST_ALIGNMENT:
    test_alignment()

if TEST_MODEL_INIT:
    test_model_init()