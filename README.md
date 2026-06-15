# Stable Diffusion From Scratch

A complete diffusion model built entirely from scratch using PyTorch, trained on CIFAR-10 with text conditioning. This project implements the core principles of Denoising Diffusion Probabilistic Models (DDPM) with a UNet backbone, multi-head self-attention, and cross-attention for text-to-image generation.

## Project Structure

```
├── dataset.py         # CIFAR-10 download, preprocessing, and text embedding
├── model.py          # UNet with time & text conditioning (built from scratch)
├── utils.py          # Noise scheduler (DDPM), image save/display helpers
├── train.py          # Training loop with checkpointing & periodic sampling
├── inference.py      # Text-to-image generation from trained checkpoint
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Architecture Overview (all from scratch)

| Component | Description |
|---|---|
| **UNet Backbone** | Encoder-decoder with skip connections, double conv blocks |
| **Time Embedding** | Sinusoidal positional encoding (Transformer-style) |
| **Text Embedding** | Learned lookup table for CIFAR-10 class names |
| **Self-Attention** | Multi-head attention at 4×4 resolution |
| **Cross-Attention** | Query from image features, Key/Value from text embedding |
| **FiLM Conditioning** | Scale & shift applied per resolution level |
| **Noise Scheduler** | Linear beta schedule (DDPM), T=1000 timesteps |

### Supported Text Prompts

The model is trained on CIFAR-10 and supports these 10 class names:

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd StableDiffusionFromScratch

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train.py --epochs 50 --batch_size 128
```

This will:
- Download CIFAR-10 automatically (if not already present)
- Train the model for 50 epochs
- Save checkpoints to `./checkpoints/`
- Generate sample images every 5 epochs in `./outputs/`
- Save the best model as `./checkpoints/best_model.pth`

**Training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 2e-4 | Learning rate |
| `--T` | 1000 | Diffusion timesteps |
| `--base_channels` | 64 | UNet base channels |
| `--text_dim` | 256 | Text embedding dimension |
| `--device` | auto | Device (`cuda` or `cpu`) |

### 2. Generate Images from Text

```bash
# Generate a single image
python inference.py --checkpoint checkpoints/best_model.pth --prompt "cat"

# Generate multiple images of the same class
python inference.py --checkpoint checkpoints/best_model.pth --prompt "dog" --num_images 4

# Generate one image per class (all 10 CIFAR-10 categories)
python inference.py --checkpoint checkpoints/best_model.pth --all_classes
```

**Inference arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Path to trained `.pth` checkpoint |
| `--prompt` | None | Text prompt (e.g., "cat", "airplane") |
| `--all_classes` | False | Generate images for all 10 classes |
| `--num_images` | 1 | Number of images for the prompt |
| `--output_dir` | ./generated | Output directory |
| `--seed` | None | Random seed for reproducibility |

## How It Works

### Training (DDPM)

1. **Forward diffusion**: Gradually add Gaussian noise to a clean image over T=1000 timesteps:
   ```
   q(x_t | x_0) = N(x_t; √(ᾱ_t)·x_0, (1-ᾱ_t)·I)
   ```

2. **Noise prediction**: Train a UNet to predict the added noise given the noisy image, timestep, and text embedding:
   ```
   L = ||ε - ε_θ(x_t, t, text)||²
   ```

### Sampling (Reverse Diffusion)

1. Start from pure Gaussian noise
2. Iteratively denoise for T steps using the predicted noise
3. The UNet is conditioned on text via cross-attention in the bottleneck

## Model Size

- Base config (64 base channels): ~14M parameters
- Runs on CPU, but GPU (CUDA) is recommended for faster training

## Results

After 50 epochs of training on CIFAR-10 (32×32 images), the model learns to generate recognizable class-conditional samples. Generated images are saved as PNG grids and individual files in `./outputs/` (during training) or `./generated/` (during inference).

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) - Ho, Jain, Abbeel (2020)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - Ronneberger et al. (2015)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - Krizhevsky (2009)

## License

MIT