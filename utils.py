"""
Utilities for the 512×512 diffusion model:

  Noise scheduler  — cosine beta schedule (better than linear for high-res)
  Prediction types — epsilon (noise) or v-prediction (more stable at high res)
  Samplers         — DDPM (stochastic) and DDIM (fast deterministic)
  Image helpers    — save grid, plot
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Cosine noise schedule  (Nichol & Dhariwal, "Improved DDPMs", 2021)
# ---------------------------------------------------------------------------
class NoiseScheduler:
    """
    Implements:
      - Cosine alpha-bar schedule
      - Forward diffusion  q(x_t | x_0)
      - DDPM reverse step
      - DDIM reverse step (fast, deterministic)
      - v-prediction support

    Usage:
        scheduler = NoiseScheduler(T=1000).to(device)
        x_t, noise = scheduler.add_noise(x0, t)
        x_prev     = scheduler.ddpm_step(model, x_t, t, text_emb, device)
        x0_approx  = scheduler.ddim_sample(model, text_emb, ...)
    """

    def __init__(
        self,
        T:           int   = 1000,
        beta_start:  float = None,      # ignored if using cosine schedule
        beta_end:    float = None,
        schedule:    str   = "cosine",  # "cosine" or "linear"
        s:           float = 0.008,     # cosine schedule offset
        prediction:  str   = "epsilon", # "epsilon" or "v"
    ):
        self.T          = T
        self.schedule   = schedule
        self.prediction = prediction

        if schedule == "cosine":
            steps = T + 1
            x     = torch.linspace(0, T, steps)
            f     = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
            alpha_bars = f / f[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clamp(betas, min=1e-5, max=0.999)
        else:
            bs = beta_start if beta_start is not None else 1e-4
            be = beta_end   if beta_end   is not None else 0.02
            betas      = torch.linspace(bs, be, T)
            alpha_bars = torch.cumprod(1.0 - betas, dim=0)

        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_alpha_bars          = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_ab        = torch.sqrt(1.0 - alpha_bars)
        self.log_one_minus_ab         = torch.log(1.0 - alpha_bars)
        self.sqrt_recip_alpha_bars    = torch.sqrt(1.0 / alpha_bars)
        self.sqrt_recip_alpha_bars_m1 = torch.sqrt(1.0 / alpha_bars - 1)

        alpha_bars_prev = torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])
        self.post_var   = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.post_var   = torch.clamp(self.post_var, min=1e-20)
        self.post_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        self.post_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alpha_bars",
            "sqrt_alpha_bars", "sqrt_one_minus_ab", "log_one_minus_ab",
            "sqrt_recip_alpha_bars", "sqrt_recip_alpha_bars_m1",
            "post_var", "post_mean_coef1", "post_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return buf[t].view(-1, 1, 1, 1)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise
        Returns (x_t, target) where target is noise or v depending on self.prediction.
        """
        noise = torch.randn_like(x0)
        x_t   = (self._gather(self.sqrt_alpha_bars, t) * x0
                 + self._gather(self.sqrt_one_minus_ab, t) * noise)
        if self.prediction == "v":
            target = (self._gather(self.sqrt_alpha_bars, t) * noise
                      - self._gather(self.sqrt_one_minus_ab, t) * x0)
            return x_t, target
        return x_t, noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if self.prediction == "v":
            return (self._gather(self.sqrt_alpha_bars, t) * x_t
                    - self._gather(self.sqrt_one_minus_ab, t) * pred)
        return (self._gather(self.sqrt_recip_alpha_bars, t) * x_t
                - self._gather(self.sqrt_recip_alpha_bars_m1, t) * pred)

    @torch.no_grad()
    def ddpm_step(self, model, x_t, t, text_emb, device, cfg_scale=7.5):
        pred = model.forward_cfg(x_t, t, text_emb, cfg_scale=cfg_scale)
        x0   = self.predict_x0(x_t, t, pred).clamp(-1, 1)
        mu   = (self._gather(self.post_mean_coef1, t) * x0
                + self._gather(self.post_mean_coef2, t) * x_t)
        noise = torch.randn_like(x_t)
        mask  = (t > 0).float().view(-1, 1, 1, 1)
        sigma = torch.sqrt(self._gather(self.post_var, t))
        return mu + mask * sigma * noise

    @torch.no_grad()
    def ddim_step(self, model, x_t, t, t_prev, text_emb, cfg_scale=7.5, eta=0.0):
        pred = model.forward_cfg(x_t, t, text_emb, cfg_scale=cfg_scale)
        x0   = self.predict_x0(x_t, t, pred).clamp(-1, 1)
        ab_t    = self._gather(self.alpha_bars, t)
        ab_prev = self._gather(self.alpha_bars, t_prev.clamp(min=0))
        sigma   = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
        dir_xt  = torch.sqrt(1 - ab_prev - sigma ** 2) * pred
        noise   = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        return torch.sqrt(ab_prev) * x0 + dir_xt + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        text_emb,
        img_size=512,
        batch_size=1,
        device=torch.device("cpu"),
        cfg_scale=7.5,
        sampler="ddim",
        ddim_steps=50,
        eta=0.0,
        show_progress=True,
    ):
        """Full reverse diffusion. Returns (B, 3, H, W) in [0, 1]."""
        model.eval()
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        if sampler == "ddim":
            step_size = max(self.T // ddim_steps, 1)
            timesteps = list(range(0, self.T, step_size))[::-1]
            pbar = tqdm(timesteps, desc="DDIM sampling", disable=not show_progress)
            for i, t_val in enumerate(pbar):
                t      = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
                x = self.ddim_step(model, x, t, t_prev, text_emb, cfg_scale, eta)
        else:
            for t_val in tqdm(reversed(range(self.T)), desc="DDPM sampling",
                              total=self.T, disable=not show_progress):
                t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
                x = self.ddpm_step(model, x, t, text_emb, device, cfg_scale)

        return (x.clamp(-1, 1) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def save_image_tensor(tensor: torch.Tensor, path: str, nrow: int = None):
    B = tensor.shape[0]
    if nrow is None:
        nrow = math.ceil(math.sqrt(B))
    ncol  = math.ceil(B / nrow)
    total = nrow * ncol
    if B < total:
        pad    = torch.zeros(total - B, *tensor.shape[1:])
        tensor = torch.cat([tensor, pad], dim=0)

    imgs = tensor.cpu().numpy().transpose(0, 2, 3, 1)
    H, W = imgs.shape[1], imgs.shape[2]
    grid = np.zeros((ncol * H, nrow * W, 3), dtype=np.float32)
    for i in range(ncol):
        for j in range(nrow):
            idx = i * nrow + j
            if idx < B:
                grid[i*H:(i+1)*H, j*W:(j+1)*W] = np.clip(imgs[idx], 0, 1)

    Image.fromarray((grid * 255).astype(np.uint8)).save(path)
    print(f"Saved image grid to {path}")


def plot_images(tensor, titles=None, figsize=(16, 4), show=True, save_path=None):
    B    = tensor.shape[0]
    ncol = min(B, 5)
    nrow = math.ceil(B / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0], figsize[1] * nrow))
    axes = np.array(axes).flatten()
    for i in range(B):
        img = np.clip(tensor[i].cpu().numpy().transpose(1, 2, 0), 0, 1)
        axes[i].imshow(img)
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=8)
        axes[i].axis("off")
    for i in range(B, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)