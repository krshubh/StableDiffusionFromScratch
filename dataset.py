"""
Dataset utilities for training a 512×512 diffusion model on MS-COCO 2017.

MS-COCO is used because:
  - 118k training images with rich human-written captions
  - Freely downloadable (~20 GB total)
  - Diverse real-world scenes → model learns meaningful text conditioning

Text encoding uses a lightweight CLIP-style tokeniser + transformer encoder
(built from scratch, no external NLP library).  The text encoder is trained
jointly with the UNet so the entire pipeline remains self-contained.

Quick start:
    python dataset.py --download          # download + verify COCO
    python dataset.py --test              # run a quick dataloader test
"""

import os
import json
import re
import math
import zipfile
import urllib.request
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MS-COCO download helpers
# ---------------------------------------------------------------------------
COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images":   "http://images.cocodataset.org/zips/val2017.zip",
    "annotations":  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

COCO_SIZES = {          # approximate compressed sizes for progress display
    "train_images": "18 GB",
    "val_images":   "1 GB",
    "annotations":  "241 MB",
}


def _reporthook(desc: str):
    pbar = [None]
    def hook(block_num, block_size, total_size):
        if pbar[0] is None:
            pbar[0] = tqdm(total=total_size, unit="B", unit_scale=True, desc=desc)
        downloaded = block_num * block_size
        pbar[0].update(min(block_size, total_size - pbar[0].n))
        if downloaded >= total_size:
            pbar[0].close()
    return hook


def download_coco(root: str = "./data/coco", splits: tuple = ("train", "val")):
    """
    Download MS-COCO 2017 images and captions.

    Directory layout after download:
        <root>/
            images/
                train2017/   (118 287 images)
                val2017/     (5 000 images)
            annotations/
                captions_train2017.json
                captions_val2017.json

    Args:
        root:   destination directory
        splits: which splits to download ("train", "val")
    """
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    keys_to_download = []
    if "train" in splits:
        keys_to_download.append("train_images")
    if "val" in splits:
        keys_to_download.append("val_images")
    keys_to_download.append("annotations")

    for key in keys_to_download:
        url = COCO_URLS[key]
        fname = url.split("/")[-1]
        dest = root / fname

        # Check if already extracted
        if key == "train_images" and (root / "images" / "train2017").exists():
            print(f"  train2017/ already extracted, skipping {fname}")
            continue
        if key == "val_images" and (root / "images" / "val2017").exists():
            print(f"  val2017/ already extracted, skipping {fname}")
            continue
        if key == "annotations" and (root / "annotations" / "captions_train2017.json").exists():
            print(f"  annotations already extracted, skipping {fname}")
            continue

        # Download
        if not dest.exists():
            print(f"Downloading {fname} (~{COCO_SIZES[key]}) ...")
            urllib.request.urlretrieve(url, dest, reporthook=_reporthook(fname))
        else:
            print(f"  {fname} already downloaded")

        # Extract
        print(f"Extracting {fname} ...")
        with zipfile.ZipFile(dest, "r") as zf:
            if key in ("train_images", "val_images"):
                zf.extractall(root / "images")
            else:
                zf.extractall(root)
        dest.unlink()   # remove zip to save space
        print(f"  {fname} extracted and zip removed.")

    print("MS-COCO 2017 ready.")


# ---------------------------------------------------------------------------
# Minimal BPE-style tokeniser (character-level fallback, no external deps)
# ---------------------------------------------------------------------------
VOCAB_SIZE   = 4096     # small but sufficient for COCO captions
MAX_SEQ_LEN  = 77       # same as CLIP for familiarity
PAD_TOKEN    = 0
BOS_TOKEN    = 1
EOS_TOKEN    = 2
UNK_TOKEN    = 3

# We build a simple character + common-subword vocabulary from COCO captions.
# If a pre-built vocab file exists we load it; otherwise we build it on first use.

_SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def _basic_tokenise(text: str) -> List[str]:
    """Split on whitespace and punctuation, lowercase."""
    text = text.lower().strip()
    text = re.sub(r"([^\w\s])", r" \1 ", text)
    return text.split()


class SimpleTokenizer:
    """
    Character-level word-piece tokenizer.  Vocabulary is built from the
    training corpus; tokens are full words for common words, individual
    characters otherwise.  No external library needed.
    """

    def __init__(self, vocab: Optional[dict] = None):
        if vocab is not None:
            self.vocab  = vocab                             # str -> int
            self.ivocab = {v: k for k, v in vocab.items()} # int -> str
        else:
            # minimal starter vocab
            self.vocab  = {t: i for i, t in enumerate(_SPECIAL_TOKENS)}
            self.ivocab = {i: t for t, i in self.vocab.items()}

    @classmethod
    def build_from_captions(cls, captions: List[str], vocab_size: int = VOCAB_SIZE) -> "SimpleTokenizer":
        from collections import Counter
        word_freq: Counter = Counter()
        for cap in captions:
            word_freq.update(_basic_tokenise(cap))

        # Most-common words first, then single chars for coverage
        words = [w for w, _ in word_freq.most_common(vocab_size - len(_SPECIAL_TOKENS) - 96)]
        chars = [chr(i) for i in range(32, 128)]   # printable ASCII

        vocab = {t: i for i, t in enumerate(_SPECIAL_TOKENS)}
        idx = len(vocab)
        for tok in chars + words:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
                if idx >= vocab_size:
                    break
        return cls(vocab)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        tokens = [BOS_TOKEN]
        for word in _basic_tokenise(text):
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # character fallback
                for ch in word:
                    tokens.append(self.vocab.get(ch, UNK_TOKEN))
        tokens.append(EOS_TOKEN)
        # Truncate / pad
        tokens = tokens[:max_len]
        tokens += [PAD_TOKEN] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        words = []
        for i in ids.tolist():
            tok = self.ivocab.get(i, "[UNK]")
            if tok == "[EOS]":
                break
            if tok not in ("[PAD]", "[BOS]"):
                words.append(tok)
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Transformer-based text encoder (built from scratch)
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """
    A small transformer encoder that maps a tokenised caption to a
    fixed-length embedding vector.

    Architecture:
        Token embedding + sinusoidal positional encoding
        → N transformer encoder layers
        → mean-pool → linear projection to text_dim
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int    = MAX_SEQ_LEN,
        d_model: int    = 512,
        n_heads: int    = 8,
        n_layers: int   = 6,
        text_dim: int   = 512,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.seq_len   = seq_len
        self.text_dim  = text_dim

        self.tok_emb   = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_emb   = self._build_pos_emb(seq_len, d_model)  # fixed
        self.drop      = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,    # pre-norm for stability
        )
        self.encoder   = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj      = nn.Linear(d_model, text_dim)
        self.norm      = nn.LayerNorm(text_dim)

    @staticmethod
    def _build_pos_emb(seq_len: int, d_model: int) -> nn.Parameter:
        pos = torch.arange(seq_len).unsqueeze(1).float()       # (T, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe  = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, T, D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, seq_len)  long tensor of token indices
        Returns:
            (B, text_dim)  sentence embedding
        """
        pad_mask = (token_ids == PAD_TOKEN)                     # (B, T) bool
        x = self.tok_emb(token_ids) + self.pos_emb             # (B, T, D)
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)     # (B, T, D)
        # Mean-pool over non-padding tokens
        mask = (~pad_mask).float().unsqueeze(-1)                # (B, T, 1)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)       # (B, D)
        return self.norm(self.proj(x))                          # (B, text_dim)


# ---------------------------------------------------------------------------
# MS-COCO Dataset
# ---------------------------------------------------------------------------
class COCOTextDataset(Dataset):
    """
    MS-COCO 2017 dataset for text-conditioned image generation.

    Returns:
        image       : (3, img_size, img_size) float32  in [-1, 1]
        token_ids   : (seq_len,) long  tokenised caption
        caption     : str  raw caption text (useful for logging)
    """

    def __init__(
        self,
        root: str         = "./data/coco",
        split: str        = "train",   # "train" | "val"
        img_size: int     = 512,
        tokenizer         = None,      # SimpleTokenizer instance
        max_samples: int  = None,      # cap dataset size (for quick tests)
        aug: bool         = True,      # use data augmentation (train only)
    ):
        root = Path(root)
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"

        self.root      = root
        self.split     = split
        self.img_size  = img_size
        self.tokenizer = tokenizer

        # ---- Load COCO captions ----
        ann_file = root / "annotations" / f"captions_{split}2017.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotations not found at {ann_file}. "
                "Run download_coco() or python dataset.py --download first."
            )
        with open(ann_file) as f:
            coco = json.load(f)

        # Build image_id → file_name lookup
        id2fname = {img["id"]: img["file_name"] for img in coco["images"]}
        img_dir  = root / "images" / f"{split}2017"

        # Each image can have 5 captions; we keep one per image (first encountered)
        seen = set()
        self.samples: List[Tuple[Path, str]] = []
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in seen:
                continue
            seen.add(img_id)
            path = img_dir / id2fname[img_id]
            if path.exists():
                self.samples.append((path, ann["caption"]))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # ---- Transforms ----
        if aug and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.12), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1,1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, caption = self.samples[idx]
        # Load image (convert greyscale / RGBA → RGB)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        # Tokenise caption
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(caption)
        else:
            token_ids = torch.zeros(MAX_SEQ_LEN, dtype=torch.long)

        return img, token_ids, caption


# ---------------------------------------------------------------------------
# Tokenizer build / load helper
# ---------------------------------------------------------------------------
def get_or_build_tokenizer(
    root: str          = "./data/coco",
    vocab_size: int    = VOCAB_SIZE,
    force_rebuild: bool = False,
) -> SimpleTokenizer:
    """
    Load tokenizer from disk if it exists, otherwise build from COCO captions.
    """
    vocab_path = Path(root) / "tokenizer_vocab.json"
    if vocab_path.exists() and not force_rebuild:
        print(f"Loading tokenizer from {vocab_path}")
        return SimpleTokenizer.load(str(vocab_path))

    print("Building tokenizer from COCO captions (this runs once)...")
    captions = []
    for split in ("train", "val"):
        ann_file = Path(root) / "annotations" / f"captions_{split}2017.json"
        if ann_file.exists():
            with open(ann_file) as f:
                data = json.load(f)
            captions.extend(a["caption"] for a in data["annotations"])

    tok = SimpleTokenizer.build_from_captions(captions, vocab_size=vocab_size)
    tok.save(str(vocab_path))
    print(f"Tokenizer saved to {vocab_path}  (vocab_size={tok.vocab_size})")
    return tok


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloader(
    root: str         = "./data/coco",
    split: str        = "train",
    img_size: int     = 512,
    batch_size: int   = 16,
    num_workers: int  = 4,
    tokenizer         = None,
    max_samples: int  = None,
    shuffle: bool     = None,
) -> DataLoader:
    dataset = COCOTextDataset(
        root=root, split=split, img_size=img_size,
        tokenizer=tokenizer, max_samples=max_samples,
    )
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download MS-COCO 2017")
    p.add_argument("--test",     action="store_true", help="Run a quick dataloader test")
    p.add_argument("--root",     type=str, default="./data/coco")
    p.add_argument("--splits",   nargs="+", default=["train", "val"])
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch",    type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.download:
        download_coco(args.root, splits=args.splits)

    if args.test:
        print("Building / loading tokenizer...")
        tok = get_or_build_tokenizer(args.root)
        print(f"  vocab_size = {tok.vocab_size}")

        print("Building dataloader...")
        loader = get_dataloader(
            root=args.root, split="train", img_size=args.img_size,
            batch_size=args.batch, num_workers=0,
            tokenizer=tok, max_samples=args.batch * 4,
        )
        imgs, token_ids, captions = next(iter(loader))
        print(f"Image batch  : {imgs.shape}       min={imgs.min():.2f} max={imgs.max():.2f}")
        print(f"Token IDs    : {token_ids.shape}")
        print(f"Sample caption: '{captions[0]}'")
        print(f"Decoded      : '{tok.decode(token_ids[0])}'")
        print("Dataset pipeline OK.")
