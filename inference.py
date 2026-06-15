"""
Inference script for the 512x512 MS-COCO diffusion model.

Loads a trained checkpoint and generates images from free-form text prompts
using DDIM sampling + Classifier-Free Guidance.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pth \
                        --prompt "a golden retriever playing in snow"

    python inference.py --checkpoint checkpoints/best_model.pth \
                        --prompt "a red sports car on a mountain road" \
                        --num_images 4 --cfg_scale 9.0 --ddim_steps 100

    python inference.py --checkpoint checkpoints/best_model.pth \
                        --prompts_file prompts.txt   # one prompt per line
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image

from dataset import get_or_build_tokenizer, TextEncoder, MAX_SEQ_LEN
from model import UNetDiffusion, EMA
from utils import NoiseScheduler, save_image_tensor, plot_images


def get_args():
    p = argparse.ArgumentParser(description="Generate 512x512 images from text")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--prompt",       default=None)
    p.add_argument("--prompts_file", default=None)
    p.add_argument("--num_images",   type=int,   default=1)
    p.add_argument("--output_dir",   default="./generated")
    p.add_argument("--device",       default=None)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--cfg_scale",    type=float, default=7.5)
    p.add_argument("--ddim_steps",   type=int,   default=50)
    p.add_argument("--eta",          type=float, default=0.0)
    p.add_argument("--sampler",      default="ddim", choices=["ddim","ddpm"])
    p.add_argument("--use_ema",      action="store_true", default=True)
    return p.parse_args()


def inference(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt       = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    print(f"  Epoch {ckpt['epoch']} | loss={ckpt['loss']:.4f}")

    tokenizer    = get_or_build_tokenizer(saved_args.data_root)

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size, seq_len=MAX_SEQ_LEN,
        d_model=saved_args.text_enc_d_model, n_heads=saved_args.text_enc_heads,
        n_layers=saved_args.text_enc_layers, text_dim=saved_args.text_dim,
    ).to(device)
    text_encoder.load_state_dict(ckpt["text_encoder_state_dict"])
    text_encoder.eval()

    model = UNetDiffusion(
        img_channels=3, base_channels=saved_args.base_channels,
        channel_mults=tuple(saved_args.channel_mults), time_dim=saved_args.time_dim,
        text_dim=saved_args.text_dim, n_heads=saved_args.n_heads,
        n_res_blocks=saved_args.n_res_blocks, dropout=0.0,
        attn_res=tuple(saved_args.attn_res), img_size=saved_args.img_size,
    ).to(device)

    if args.use_ema and "ema_state_dict" in ckpt:
        print("  Loading EMA weights")
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    scheduler = NoiseScheduler(
        T=saved_args.T, schedule=saved_args.schedule, prediction=saved_args.prediction,
    ).to(device)

    # Build prompt list
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    elif args.prompt:
        prompts = [args.prompt] * args.num_images
    else:
        raise ValueError("Provide --prompt or --prompts_file")

    print(f"Generating {len(prompts)} image(s)  "
          f"[cfg={args.cfg_scale}, {args.sampler} steps={args.ddim_steps}, eta={args.eta}]")

    os.makedirs(args.output_dir, exist_ok=True)
    all_images  = []
    all_prompts = []

    chunk = 4
    for i in range(0, len(prompts), chunk):
        batch_prompts = prompts[i:i+chunk]
        token_ids = torch.stack([tokenizer.encode(p) for p in batch_prompts]).to(device)

        with torch.no_grad():
            text_emb = text_encoder(token_ids)
            samples  = scheduler.sample(
                model=model, text_emb=text_emb,
                img_size=saved_args.img_size, batch_size=len(batch_prompts),
                device=device, cfg_scale=args.cfg_scale,
                sampler=args.sampler, ddim_steps=args.ddim_steps,
                eta=args.eta, show_progress=True,
            )

        all_images.extend([samples[j] for j in range(len(batch_prompts))])
        all_prompts.extend(batch_prompts)

    if not all_images:
        print("No images generated.")
        return

    images_tensor = torch.stack(all_images, dim=0)

    grid_path = os.path.join(args.output_dir, "generated_grid.png")
    save_image_tensor(images_tensor, grid_path, nrow=min(4, len(images_tensor)))

    for idx, (img, prompt) in enumerate(zip(images_tensor, all_prompts)):
        img_np   = np.clip(img.cpu().numpy().transpose(1, 2, 0), 0, 1)
        pil_img  = Image.fromarray((img_np * 255).astype(np.uint8))
        safe     = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:60]
        out_path = os.path.join(args.output_dir, f"{idx:03d}_{safe}.png")
        pil_img.save(out_path)
        print(f"  Saved: {out_path}")

    plot_images(
        images_tensor, titles=[p[:50] for p in all_prompts],
        show=False, save_path=os.path.join(args.output_dir, "generated_plot.png"),
    )
    print(f"\nDone! {len(all_images)} image(s) in '{args.output_dir}/'")


if __name__ == "__main__":
    args = get_args()
    inference(args)