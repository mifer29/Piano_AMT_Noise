import os
import sys
import math
import time
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Adafactor

from maestro_dataset import MAESTROSeq2SeqDataset, collate_fn
from model import AudioTransformer, shift_tokens_right



parser = argparse.ArgumentParser()
parser.add_argument("--maestro_root", type=str, required=True)
parser.add_argument("--events_dir", type=str, required=True)
parser.add_argument("--mel_dir", type=str, required=True)
parser.add_argument("--output_root", type=str, required=True)
parser.add_argument("--total_steps", type=int, default=500)
args = parser.parse_args()




timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(args.output_root, f"overfit_{timestamp}")
os.makedirs(RUN_DIR, exist_ok=True)
LOG_FILE = os.path.join(RUN_DIR, "overfit.log")


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

log('OVERFITTING ONE BATCH TEST')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
log(f"GPUs: {torch.cuda.device_count()}")



dataset = MAESTROSeq2SeqDataset(
    maestro_path=args.maestro_root,
    events_dir=args.events_dir,
    mel_dir=args.mel_dir,
    split="train",
    logger=log
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

# Pull a single batch and freeze it — we'll overfit only this
batch = next(iter(loader))
audio = batch["audio_features"].to(device)   # [B, T, n_mels]
targets = batch["target_ids"].to(device)     # [B, L]

log(f"Batch audio shape:   {audio.shape}")
log(f"Batch targets shape: {targets.shape}")
log(f"Audio mean: {audio.mean():.4f}, std: {audio.std():.4f}")
log(f"Audio min:  {audio.min():.4f},  max: {audio.max():.4f}")
log(f"Has NaN in audio:   {torch.isnan(audio).any().item()}")
log(f"Has NaN in targets: {torch.isnan(targets.float()).any().item()}")
log(f"Target sample[0] (first 30): {targets[0][:30].tolist()}")
log(f"Unique tokens in batch: {targets.unique().tolist()[:30]}")
log(f"Pad fraction: {(targets == dataset.pad_id).float().mean().item():.3f}")
log(f"Vocab size: {len(dataset.vocab)}")
log(f"BOS id: {dataset.bos_id}, EOS id: {dataset.eos_id}, PAD id: {dataset.pad_id}")




model = AudioTransformer(
    n_mels=512,
    vocab_size=len(dataset.vocab)
).to(device)

total_params = sum(p.numel() for p in model.parameters())
log(f"Model parameters: {total_params / 1e6:.1f}M")



optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    scale_parameter=False,
    relative_step=False,
    warmup_init=False
)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_id)




log("\nStarting overfit loop")
log(f"{'Step':>6} | {'Loss':>10} | {'PPL':>12} | {'Grad Norm':>10}")


start = time.time()

for step in range(1, args.total_steps + 1):

    model.train()

    decoder_input = shift_tokens_right(targets, dataset.bos_id)
    logits = model(audio, decoder_input)

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    # Guard against NaN
    if not torch.isfinite(loss):
        log(f" Non-finite loss at step {step}: {loss.item()}. Aborting.")
        sys.exit(1)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    ppl = math.exp(min(loss.item(), 20))

    if step % 10 == 0 or step == 1:
        elapsed = (time.time() - start) / 60
        log(
            f"Step {step:>4}/{args.total_steps} | "
            f"Loss: {loss.item():>8.4f} | "
            f"PPL: {ppl:>10.2f} | "
            f"Grad norm: {grad_norm:>8.4f} | "
            f"Time: {elapsed:.1f} min"
        )



final_loss = loss.item()
log(f"Final loss after {args.total_steps} steps: {final_loss:.4f}")

if final_loss < 0.5:
    log(" PASSED, model can overfit a single batch.")
elif final_loss < 2.0:
    log(" PARTIAL, loss dropped but not near zero. May need more steps.")
else:
    log(" FAILED, loss did not decrease enough. Model cannot overfit.")
