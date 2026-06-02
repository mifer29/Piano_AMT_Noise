import os
import sys
import math
import time
import argparse
import datetime
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adafactor


from maestro_dataset import MAESTROSeq2SeqDataset, collate_fn
from model import AudioTransformer, shift_tokens_right, greedy_decode

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def get_gpu_util(device_index=0):
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    except Exception:
        return None


def train():

    # Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_train",        type=str, required=True)
    parser.add_argument("--hdf5_val",          type=str, required=True)
    parser.add_argument("--output_root",       type=str, required=True)
    parser.add_argument("--batch_size",        type=int, default=16)
    parser.add_argument("--total_steps",       type=int, default=50000)
    parser.add_argument("--max_output_tokens", type=int, default=512)
    parser.add_argument("--resume_run", type=str, default=None)
    args = parser.parse_args()

    # Experiment folder and logging
    if args.resume_run is not None:
        RUN_DIR = args.resume_run
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_DIR = os.path.join(args.output_root, f"run_{timestamp}")

    os.makedirs(RUN_DIR, exist_ok=True)

    CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    LOG_FILE     = os.path.join(RUN_DIR, "train.log")
    metrics_path = os.path.join(RUN_DIR, "metrics.pt")

    def log(msg):
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
        print(line, flush=True)


    log("NEW RUN STARTED")
    log(f"Run directory:  {RUN_DIR}")
    log(f"Arguments:      {vars(args)}")
    log(f"Python:         {sys.version}")
    log(f"PyTorch:        {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA version:   {torch.version.cuda}")
    log(f"GPU count:      {torch.cuda.device_count()}")
    log(f"pynvml:         {'available' if NVML_AVAILABLE else 'not available'}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"  GPU {i}: {props.name}, VRAM: {props.total_memory / 1e9:.1f}GB")

    # --------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------

    GRAD_CLIP          = 1.0
    LOG_EVERY          = 50
    VALIDATE_EVERY     = 500
    SAVE_EVERY         = 50000
    RESUME_SAVE_EVERY  = 1000
    GREEDY_EVERY       = 2000
    NUM_GREEDY_SAMPLES = 2
    WARMUP_STEPS       = 1000

    log(f"GRAD_CLIP:         {GRAD_CLIP}")
    log(f"WARMUP_STEPS:      {WARMUP_STEPS}")
    log(f"BATCH_SIZE:        {args.batch_size}")
    log(f"TOTAL_STEPS:       {args.total_steps}")
    log(f"MAX_OUTPUT_TOKENS: {args.max_output_tokens}")

    # Metrics lists: populated throughout training
    train_losses = []
    val_losses   = []
    train_steps  = []
    val_steps    = []

    def save_metrics():
        torch.save({
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "train_steps":  train_steps,
            "val_steps":    val_steps,
        }, metrics_path)

    # Device

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    log(f"Using device: {device}, num_gpus: {num_gpus}")

    # Dataset

    log("Loading train dataset")
    try:
        train_dataset = MAESTROSeq2SeqDataset(
            hdf5_path=args.hdf5_train,
            split="train",
            max_output_tokens=args.max_output_tokens,
            logger=log
        )
        log(f"Train dataset loaded: {len(train_dataset)} samples")
        log(f"Vocab size: {len(train_dataset.vocab)}")
    except Exception as e:
        log(f"FATAL: Failed to load train dataset: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    log("Loading val dataset")
    try:
        val_dataset = MAESTROSeq2SeqDataset(
            hdf5_path=args.hdf5_val,
            split="validation",
            max_output_tokens=args.max_output_tokens,
            logger=log
        )
        log(f"Val dataset loaded: {len(val_dataset)} samples")
    except Exception as e:
        log(f"FATAL: Failed to load val dataset: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    # DataLoaders

    log("Creating DataLoaders")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        log("DataLoaders created OK")
    except Exception as e:
        log(f"FATAL: Failed to create DataLoaders: {e}")
        log(traceback.format_exc())
        sys.exit(1)


    # Test one batch before doing anything else


    log("Testing first batch load")
    try:
        t0         = time.time()
        test_batch = next(iter(train_loader))
        log(f"First batch loaded in {time.time()-t0:.2f}s")
        log(f"  audio_features shape: {test_batch['audio_features'].shape}")
        log(f"  target_ids shape:     {test_batch['target_ids'].shape}")
        log(f"  audio dtype:          {test_batch['audio_features'].dtype}")
        log(f"  audio min/max:        {test_batch['audio_features'].min():.3f} / {test_batch['audio_features'].max():.3f}")
        log(f"  target min/max:       {test_batch['target_ids'].min()} / {test_batch['target_ids'].max()}")
    except Exception as e:
        log(f"FATAL: First batch failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)


    # Model

    log("Creating model")
    try:
        model = AudioTransformer(
            n_mels=512,
            vocab_size=len(train_dataset.vocab)
        )
        total_params = sum(p.numel() for p in model.parameters())
        log(f"Model created: {total_params/1e6:.1f}M parameters")
    except Exception as e:
        log(f"FATAL: Model creation failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    if num_gpus > 1:
        log(f"Wrapping with DataParallel on {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    log("Moving model to device...")
    try:
        model = model.to(device)
        log(f"Model on device OK, GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
    except Exception as e:
        log(f"FATAL: model.to(device) failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)

    
    # Test forward pass

    log("Testing forward pass")
    try:
        model.eval()
        with torch.no_grad():
            audio         = test_batch["audio_features"].to(device)
            targets       = test_batch["target_ids"].to(device)
            decoder_input = shift_tokens_right(targets, train_dataset.bos_id)
            logits        = model(audio, decoder_input)
            log(f"Forward pass OK, logits shape: {logits.shape}")
            log(f"GPU memory after forward pass: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
        model.train()
    except Exception as e:
        log(f"FATAL: Forward pass test failed: {e}")
        log(traceback.format_exc())
        log(f"  GPU allocated: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
        log(f"  GPU reserved:  {torch.cuda.memory_reserved(device)/1e9:.2f}GB")
        sys.exit(1)

 
    # Optimizer + scheduler
    

    log("Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    def warmup_lr(step):
        return min((step + 1) / WARMUP_STEPS, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)

    log("Optimizer: AdamW lr=1e-4, betas=(0.9,0.98), wd=0.01")
    log(f"Scheduler: linear warmup over {WARMUP_STEPS} steps")


    # Resume

    resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")
    global_step = 0

    if os.path.exists(resume_path):
        log(f"Resuming from {resume_path}...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # Restore metrics history so curves are continuous after resume
            if "train_losses" in checkpoint:
                train_losses.extend(checkpoint["train_losses"])
                train_steps.extend(checkpoint["train_steps"])
                val_losses.extend(checkpoint["val_losses"])
                val_steps.extend(checkpoint["val_steps"])
            global_step = checkpoint["step"]
            log(f"Resumed from step {global_step}")
            log(f"  Restored {len(train_losses)} train / {len(val_losses)} val loss entries")
        except Exception as e:
            log(f"WARNING: Resume failed, starting from scratch: {e}")
            global_step = 0
    else:
        log("No checkpoint found, starting from scratch")

   
    # Validation helpers
  
    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = 0.0
        for batch in val_loader:
            audio         = batch["audio_features"].to(device, non_blocking=True)
            targets       = batch["target_ids"].to(device, non_blocking=True)
            decoder_input = shift_tokens_right(targets, train_dataset.bos_id)
            logits        = model(audio, decoder_input)
            loss          = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss   += loss.item()
        model.train()
        return total_loss / max(len(val_loader), 1)

    @torch.no_grad()
    def evaluate_greedy():
        model.eval()
        log("Running greedy evaluation")
        model_for_decode = model.module if hasattr(model, "module") else model
        for i, batch in enumerate(val_loader):
            if i >= NUM_GREEDY_SAMPLES:
                break
            audio       = batch["audio_features"][0:1].to(device)
            target      = batch["target_ids"][0]
            generated   = greedy_decode(
                model_for_decode, audio,
                bos_id=train_dataset.bos_id,
                eos_id=train_dataset.eos_id,
                max_len=target.size(0)
            )
            target_list = [t for t in target.tolist() if t != train_dataset.pad_id]
            min_len     = min(len(generated), len(target_list))
            correct     = sum(g == t for g, t in zip(generated[:min_len], target_list[:min_len]))
            acc         = correct / max(min_len, 1)
            log(f"Greedy sample {i} | token acc: {acc:.4f}")
            log(f"  Target (first 20):    {target_list[:20]}")
            log(f"  Generated (first 20): {generated[:20]}")
        model.train()


    # Gradient accumulation
    GRAD_ACCUM_STEPS = 256 // args.batch_size  # e.g. 8
    log(f"GRAD_ACCUM_STEPS: {GRAD_ACCUM_STEPS}")

    log("TRAINING STARTED")

    start_time = time.time()
    t_step_end = time.time()

    optimizer.zero_grad()

    while global_step < args.total_steps:

        model.train()

        for batch in train_loader:

            t_data_ready = time.time()

            try:
                audio   = batch["audio_features"].to(device, non_blocking=True)
                targets = batch["target_ids"].to(device, non_blocking=True)
                t_transfer = time.time()

                decoder_input = shift_tokens_right(targets, train_dataset.bos_id)
                logits = model(audio, decoder_input)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )

                if not torch.isfinite(loss):
                    log(f"FATAL: Non-finite loss {loss.item()} at step {global_step}")
                    sys.exit(1)

                # IMPORTANT: scale loss for accumulation
                loss_scaled = loss / GRAD_ACCUM_STEPS

                loss_scaled.backward()

                t_backward = time.time()

            except RuntimeError as e:
                log(f"FATAL: RuntimeError at step {global_step}: {e}")
                log(traceback.format_exc())
                sys.exit(1)

            # Only update every accumulation steps
            if (global_step + 1) % GRAD_ACCUM_STEPS == 0:

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]

            # ---------------- LOGGING ----------------
            if global_step <= 10 or global_step % LOG_EVERY == 0:

                ppl = math.exp(min(loss.item(), 20))  # use original loss
                steps_per_s = global_step / (time.time() - start_time)
                eta_hours = (args.total_steps - global_step) / max(steps_per_s, 1e-6) / 3600
                mem_alloc = torch.cuda.memory_allocated(device) / 1e9

                log(
                    f"Step {global_step}/{args.total_steps} | "
                    f"Loss: {loss.item():.4f} | PPL: {ppl:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"steps/s: {steps_per_s:.2f} | "
                    f"ETA: {eta_hours:.1f}h | "
                    f"GPU mem: {mem_alloc:.2f}GB"
                )

                train_losses.append(loss.item())
                train_steps.append(global_step)

            t_step_end = time.time()

            # ---------------- VALIDATION ----------------
            if global_step % VALIDATE_EVERY == 0:
                val_loss = evaluate()
                val_losses.append(val_loss)
                val_steps.append(global_step)
                log(f"Validation @ step {global_step} | Val Loss: {val_loss:.4f}")
                save_metrics()

            if global_step % GREEDY_EVERY == 0:
                evaluate_greedy()

            # ---------------- CHECKPOINT ----------------
            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "train_steps": train_steps,
                    "val_losses": val_losses,
                    "val_steps": val_steps,
                }, ckpt_path)
                log(f"Checkpoint saved: {ckpt_path}")

            if global_step % RESUME_SAVE_EVERY == 0:
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "train_steps": train_steps,
                    "val_losses": val_losses,
                    "val_steps": val_steps,
                }, resume_path)
                log(f"Resume checkpoint saved at step {global_step}")

            if global_step >= args.total_steps:
                break

    # End training

    log("TRAINING FINISHED")
    log(f"Total time: {(time.time()-start_time)/3600:.2f}h")

    save_metrics()
    log(f"Final metrics saved to {metrics_path}")
    log(f"  {len(train_losses)} train loss entries")
    log(f"  {len(val_losses)} val loss entries")

    torch.save({
        "step":                 global_step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses":         train_losses,
        "train_steps":          train_steps,
        "val_losses":           val_losses,
        "val_steps":            val_steps,
    }, resume_path)
    log("Final checkpoint saved")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)