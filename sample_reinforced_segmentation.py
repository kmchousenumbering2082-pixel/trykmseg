#!/usr/bin/env python3
"""Train a lightweight reinforced segmentation model on local image-mask pairs.

This script is intentionally self-contained so it can be run directly on the
uploaded `sample_images/` and `sample_masks/` folders.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tifffile import imread, imwrite
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class PairPath:
    image: Path
    mask: Path


class PairSegmentationDataset(Dataset):
    def __init__(self, pairs: Sequence[PairPath]):
        self.pairs = list(pairs)

        all_classes = set()
        for pair in self.pairs:
            all_classes.update(np.unique(imread(str(pair.mask))).tolist())
        self.class_values = sorted(int(c) for c in all_classes)
        self.class_to_index = {v: i for i, v in enumerate(self.class_values)}

    def __len__(self) -> int:
        return len(self.pairs)

    def _to_tensor_image(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.astype(np.float32)
        arr -= arr.min()
        max_val = arr.max() if arr.max() > 0 else 1.0
        arr /= max_val
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def _to_tensor_mask(self, arr: np.ndarray) -> torch.Tensor:
        mapped = np.zeros_like(arr, dtype=np.int64)
        for raw_value, idx in self.class_to_index.items():
            mapped[arr == raw_value] = idx
        return torch.from_numpy(mapped)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        image = self._to_tensor_image(imread(str(pair.image)))
        mask = self._to_tensor_mask(imread(str(pair.mask)))
        return image, mask, idx


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.enc1 = self.block(in_channels, 32)
        self.enc2 = self.block(32, 64)
        self.bottleneck = self.block(64, 128)
        self.dec2 = self.block(128 + 64, 64)
        self.dec1 = self.block(64 + 32, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


class ReinforcedSampler:
    """Bandit-like sampler that increases probability of hard/rewarding examples."""

    def __init__(self, dataset_size: int, temperature: float = 0.6, momentum: float = 0.9):
        self.logits = torch.zeros(dataset_size, dtype=torch.float32)
        self.temperature = temperature
        self.momentum = momentum

    def sample(self, k: int) -> torch.Tensor:
        probs = torch.softmax(self.logits / self.temperature, dim=0)
        return torch.multinomial(probs, num_samples=k, replacement=False)

    def update(self, indices: torch.Tensor, rewards: torch.Tensor) -> None:
        for idx, reward in zip(indices.tolist(), rewards.tolist()):
            old = self.logits[idx].item()
            self.logits[idx] = self.momentum * old + (1 - self.momentum) * reward


def load_pairs(images_dir: Path, masks_dir: Path) -> List[PairPath]:
    image_paths = sorted(images_dir.glob("*.tif"))
    pairs: List[PairPath] = []
    for image_path in image_paths:
        mask_path = masks_dir / image_path.name
        if mask_path.exists():
            pairs.append(PairPath(image=image_path, mask=mask_path))
    if not pairs:
        raise RuntimeError("No matching image-mask *.tif pairs found.")
    return pairs


def mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            ious.append(intersection / union)
    return torch.stack(ious).mean() if ious else torch.tensor(0.0, device=logits.device)


def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_pairs(Path(args.images_dir), Path(args.masks_dir))
    dataset = PairSegmentationDataset(pairs)

    val_count = max(1, int(len(dataset) * args.val_ratio))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_indices, train_indices = indices[:val_count], indices[val_count:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    in_channels = dataset[0][0].shape[0]
    model = TinyUNet(in_channels, len(dataset.class_values)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    sampler = ReinforcedSampler(dataset_size=len(train_indices), temperature=args.temp)
    baseline_reward = 0.0

    train_loader_full = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    history = []
    for epoch in range(args.epochs):
        model.train()
        sampled_local_indices = sampler.sample(k=min(len(train_indices), args.steps_per_epoch))
        sampled_global_indices = [train_indices[i] for i in sampled_local_indices.tolist()]
        sampled_subset = Subset(dataset, sampled_global_indices)
        sampled_loader = DataLoader(sampled_subset, batch_size=args.batch_size, shuffle=True)

        epoch_losses = []
        epoch_rewards = []
        for images, masks, _ in sampled_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                batch_iou = mean_iou(logits, masks, len(dataset.class_values)).item()
            reward = batch_iou - baseline_reward
            baseline_reward = 0.95 * baseline_reward + 0.05 * batch_iou
            epoch_losses.append(loss.item())
            epoch_rewards.append(reward)

        if epoch_rewards:
            update_rewards = torch.full_like(sampled_local_indices, float(np.mean(epoch_rewards)), dtype=torch.float32)
            sampler.update(sampled_local_indices, update_rewards)

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_iou += mean_iou(logits, masks, len(dataset.class_values)).item()
        val_iou /= max(1, len(val_loader))
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "val_miou": val_iou,
        })
        print(f"Epoch {epoch+1}/{args.epochs} - loss={history[-1]['train_loss']:.4f} val_mIoU={val_iou:.4f}")

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for images, _, idx in pred_loader:
            images = images.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            original_values = np.array(dataset.class_values, dtype=np.uint8)
            restored = original_values[pred]
            file_name = pairs[int(idx.item())].image.name
            imwrite(str(pred_dir / file_name), restored)

    metrics = {
        "num_pairs": len(dataset),
        "classes": dataset.class_values,
        "history": history,
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), output_dir / "tiny_unet_reinforced.pth")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reinforced segmentation on local image-mask pairs")
    parser.add_argument("--images-dir", default="sample_images", help="Directory with input .tif images")
    parser.add_argument("--masks-dir", default="sample_masks", help="Directory with label .tif masks")
    parser.add_argument("--output-dir", default="outputs/reinforced_segmentation", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
