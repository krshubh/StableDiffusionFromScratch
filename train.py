"""
Memory-optimized training script for 15 GB VRAM.

Changes from original train.py:
  - img_size=256 instead of 512 (4x fewer pixels → 4x less activation memory)
  - base_channels=64, attn_res=(32,16,8) — no 64x64 attention
  - time_dim=text_dim=256 (half of original)
  - batch_size=2, grad_accum=8 → effective batch=16 with minimal peak VRAM
  - text_enc_d_model=256, text_enc_layers=4 (smaller text encoder)
  - EMA on CPU (model.py change) — saves ~model_size VRAM
  - gradient checkpointing (model.py change) — saves ~40% activation memory
  - torch.backends.cuda.matmul.allow_tf32 = True — faster matmul on Ampere+
  - Empty CUDA cache before validation sampling
  - autocast device_type inferred at runtime (works for CPU too)

To train on a single 15 GB GPU:
    python train.py

To resume:
    python train.py --resume checkpoints/best_model.pth
"""

import os
import argparse
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from dataset import get_dataloader, get_or_build_tokenizer, TextEncoder, MAX_SEQ_LEN
from model import UNetDiffusion, EMA
from utils import NoiseScheduler, save_image_tensor, plot_images


def get_args():
    p = argparse.ArgumentParser(description="Train diffusion model on 15 GB VRAM")

    # Paths
    p.add_argument("--data_root",    default="./data/coco")
    p.add_argument("--ckpt_dir",     default="./checkpoints")
    p.add_argument("--output_dir",   default="./outputs")
    p.add_argument("--resume",       default=None)

    # ---- Model: tuned for 15 GB VRAM ----
    p.add_argument("--img_size",      type=int,   default=256,
                   help="256 fits 15 GB; use 512 only with A100+")
    p.add_argument("--base_channels", type=int,   default=64)
    p.add_argument("--channel_mults", type=int,   nargs="+", default=[1, 2, 4, 4])
    p.add_argument("--time_dim",      type=int,   default=256)
    p.add_argument("--text_dim",      type=int,   default=256)
    p.add_argument("--n_heads",       type=int,   default=8)
    p.add_argument("--n_res_blocks",  type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--attn_res",      type=int,   nargs="+", default=[32, 16, 8],
                   help="No 64x64 attn — that alone saves ~3 GB")
    p.add_argument("--use_checkpoint", action="store_true", default=True,
                   help="Gradient checkpointing (saves ~40%% activation memory)")

    # ---- Text encoder: smaller for memory ----
    p.add_argument("--text_enc_layers",  type=int, default=4)
    p.add_argument("--text_enc_d_model", type=int, default=256)
    p.add_argument("--text_enc_heads",   type=int, default=4)

    # ---- Diffusion ----
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--schedule",   default="cosine",  choices=["cosine", "linear"])
    p.add_argument("--prediction", default="epsilon", choices=["epsilon", "v"])
    p.add_argument("--cfg_prob",   type=float, default=0.15)

    # ---- Training: tuned for 15 GB VRAM ----
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=2,
                   help="Per-GPU micro batch. Keep at 2 for 15 GB.")
    p.add_argument("--grad_accum",    type=int,   default=8,
                   help="Effective batch = batch_size * grad_accum = 16")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--warmup_steps",  type=int,   default=2000)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--amp",           action="store_true", default=True,
                   help="fp16 mixed precision — halves activation memory")
    p.add_argument("--ema_decay",     type=float, default=0.9999)

    # ---- Logging ----
    p.add_argument("--device",          default=None)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--log_interval",    type=int, default=50)
    p.add_argument("--save_interval",   type=int, default=5)
    p.add_argument("--sample_interval", type=int, default=5)
    p.add_argument("--ddim_steps",      type=int, default=50)
    p.add_argument("--cfg_scale",       type=float, default=7.5)
    p.add_argument("--num_val_samples", type=int, default=4)

    return p.parse_args()


def get_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save(path, epoch, step, model, text_encoder, optimizer, ema, loss, best_loss, args):
    torch.save({
        "epoch":                   epoch,
        "global_step":             step,
        "model_state_dict":        model.state_dict(),
        "text_encoder_state_dict": text_encoder.state_dict(),
        "optimizer_state_dict":    optimizer.state_dict(),
        "ema_state_dict":          ema.state_dict(),
        "loss":                    loss,
        "best_loss":               best_loss,
        "args":                    args,
    }, path)


@torch.no_grad()
def _generate_samples(model, text_encoder, scheduler, tokenizer,
                      val_loader, args, device, epoch):
    # Free any cached activations before sampling (important on 15 GB)
    torch.cuda.empty_cache()

    model.eval()
    text_encoder.eval()

    images_gt, token_ids, captions = next(iter(val_loader))
    token_ids = token_ids[:args.num_val_samples].to(device)
    captions  = captions[:args.num_val_samples]

    with autocast(device.type, enabled=args.amp):
        text_emb = text_encoder(token_ids)

    print(f"\nGenerating {args.num_val_samples} samples (DDIM {args.ddim_steps} steps)...")

    # Use EMA weights: they live on CPU, get loaded to GPU here temporarily
    with ema.apply(model):
        samples = scheduler.sample(
            model=model, text_emb=text_emb,
            img_size=args.img_size, batch_size=args.num_val_samples,
            device=device, cfg_scale=args.cfg_scale,
            sampler="ddim", ddim_steps=args.ddim_steps, show_progress=True,
        )

    save_image_tensor(samples, os.path.join(args.output_dir, f"epoch_{epoch:04d}_grid.png"))
    plot_images(samples, titles=[c[:40] for c in captions], show=False,
                save_path=os.path.join(args.output_dir, f"epoch_{epoch:04d}_plot.png"))

    torch.cuda.empty_cache()
    model.train()
    text_encoder.train()


def train(args):
    # ---- Device ----
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Performance flags ----
    if device.type == "cuda":
        # TF32 on Ampere (3090, A10, etc.): faster matmul with minimal precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.ckpt_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Tokenizer + DataLoaders ----
    tokenizer    = get_or_build_tokenizer(args.data_root)
    train_loader = get_dataloader(
        root=args.data_root, split="train", img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers, tokenizer=tokenizer,
    )
    val_loader = get_dataloader(
        root=args.data_root, split="val", img_size=args.img_size,
        batch_size=args.num_val_samples, num_workers=1, tokenizer=tokenizer,
        max_samples=args.num_val_samples * 10, shuffle=False,
    )
    print(f"Train  : {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    # ---- Text encoder (small) ----
    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size, seq_len=MAX_SEQ_LEN,
        d_model=args.text_enc_d_model, n_heads=args.text_enc_heads,
        n_layers=args.text_enc_layers, text_dim=args.text_dim,
    ).to(device)

    # ---- UNet ----
    model = UNetDiffusion(
        img_channels=3,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        time_dim=args.time_dim,
        text_dim=args.text_dim,
        n_heads=args.n_heads,
        n_res_blocks=args.n_res_blocks,
        dropout=args.dropout,
        attn_res=tuple(args.attn_res),
        img_size=args.img_size,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    unet_params = sum(p.numel() for p in model.parameters())
    text_params = sum(p.numel() for p in text_encoder.parameters())
    print(f"UNet   : {unet_params/1e6:.1f}M params")
    print(f"Text   : {text_params/1e6:.1f}M params")
    print(f"Total  : {(unet_params+text_params)/1e6:.1f}M params")

    # ---- EMA (CPU) ----
    ema = EMA(model, decay=args.ema_decay)   # weights on CPU — saves GPU VRAM

    # ---- Noise scheduler ----
    scheduler = NoiseScheduler(
        T=args.T, schedule=args.schedule, prediction=args.prediction
    ).to(device)

    # ---- Optimiser ----
    all_params = list(model.parameters()) + list(text_encoder.parameters())
    optimizer  = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.999), eps=1e-8)
    # fp16 scaler — only active when device is cuda
    scaler = GradScaler(device.type, enabled=(args.amp and device.type == "cuda"))

    # ---- Resume ----
    start_epoch = 1
    global_step = 0
    best_loss   = float("inf")
    avg_loss    = float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        text_encoder.load_state_dict(ckpt["text_encoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss   = ckpt.get("best_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    total_steps = len(train_loader) * args.epochs // max(args.grad_accum, 1)
    t0 = time.time()

    print(f"\nEffective batch size : {args.batch_size * args.grad_accum}")
    print(f"Gradient checkpointing: {args.use_checkpoint}")
    print(f"Mixed precision (AMP) : {args.amp}")
    print(f"EMA on CPU            : True")
    print(f"Starting training...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        text_encoder.train()
        epoch_losses = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step_i, (images, token_ids, _) in enumerate(pbar):
            images    = images.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)

            # --- Text encode under AMP ---
            with autocast(device.type, enabled=(args.amp and device.type == "cuda")):
                text_emb = text_encoder(token_ids)

            # --- CFG dropout: replace text with null embedding ---
            if random.random() < args.cfg_prob:
                text_emb = model.null_text.expand(images.shape[0], -1).detach()

            # --- Random timesteps ---
            t = torch.randint(0, args.T, (images.shape[0],), device=device).long()

            # --- Forward + loss under AMP ---
            with autocast(device.type, enabled=(args.amp and device.type == "cuda")):
                x_noisy, target = scheduler.add_noise(images, t)
                pred             = model(x_noisy, t, text_emb)
                loss             = nn.functional.mse_loss(pred, target) / args.grad_accum

            # --- Backward ---
            scaler.scale(loss).backward()

            # --- Optimizer step every grad_accum mini-batches ---
            if (step_i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)   # set_to_none saves memory vs zero

                # LR schedule
                lr = get_lr(global_step, args.warmup_steps, total_steps, args.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # EMA update runs on CPU — no GPU memory cost
                ema.update(model)
                global_step += 1

            loss_val = loss.item() * args.grad_accum
            epoch_losses.append(loss_val)
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
            })

            if global_step > 0 and global_step % args.log_interval == 0:
                window = args.log_interval * args.grad_accum
                avg = float(np.mean(epoch_losses[-window:]))
                print(f"  step={global_step:6d} | loss={avg:.4f} | "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                      f"t={time.time()-t0:.0f}s")

        avg_loss = float(np.mean(epoch_losses))
        print(f"--- Epoch {epoch} | avg_loss={avg_loss:.4f} ---")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save(os.path.join(args.ckpt_dir, "best_model.pth"),
                  epoch, global_step, model, text_encoder,
                  optimizer, ema, avg_loss, best_loss, args)
            print(f"  Best model saved (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            _save(os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pth"),
                  epoch, global_step, model, text_encoder,
                  optimizer, ema, avg_loss, best_loss, args)

        # Validation sampling
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            _generate_samples(model, text_encoder, scheduler, tokenizer,
                               val_loader, args, device, epoch)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min | best_loss={best_loss:.4f}")
    _save(os.path.join(args.ckpt_dir, "final_model.pth"),
          args.epochs, global_step, model, text_encoder,
          optimizer, ema, avg_loss, best_loss, args)


if __name__ == "__main__":
    args = get_args()
    train(args)
