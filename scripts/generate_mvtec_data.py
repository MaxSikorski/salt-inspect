#!/usr/bin/env python3
"""
Generate all precomputed data for the SALT Industrial Inspector website.

Trains SALT on MVTec AD (224×224 industrial images), exports ONNX model,
and generates all assets for the in-browser anomaly detection demo.

Outputs:
  1. salt-inspector.onnx          — Exported ViT student model (224×224 input)
  2. training-metrics.json        — Per-step loss curves for both SALT stages
  3. mvtec-samples.json           — Sample images per category as base64
  4. reference-embeddings.json    — Per-category normal reference embeddings

Usage (local with GPU):
    python scripts/generate_mvtec_data.py --output-dir site/

Usage (Colab):
    See notebooks/train_mvtec_colab.ipynb
"""

import base64
import copy
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.data_utils import get_image_transforms
from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import build_encoder
from src.ijepa.models.predictor import build_predictor
from src.salt.models.mae_decoder import build_mae_decoder
from src.salt.losses.pixel_loss import PixelReconstructionLoss
from src.salt.train_stage1 import SALTStage1Trainer
from src.salt.train_stage2 import SALTStage2Trainer

# ---- MVTec AD Configuration ----
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Model config: ViT for 224×224 images, patch_size=16 → 14×14 = 196 patches
ENC_CONFIG = {
    "img_size": 224,
    "patch_size": 16,
    "embed_dim": 192,
    "depth": 12,
    "num_heads": 3,
    "mlp_ratio": 4.0,
}

DEC_CONFIG = {
    "encoder_embed_dim": 192,
    "decoder_embed_dim": 96,
    "decoder_depth": 4,
    "decoder_num_heads": 3,
    "patch_size": 16,
    "in_channels": 3,
}

PRED_CONFIG = {
    "embed_dim": 192,
    "predictor_embed_dim": 96,
    "depth": 4,
    "num_heads": 3,
}

# Training hyperparameters
IMG_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = 14  # 224 / 16 = 14
NUM_PATCHES = 196  # 14 * 14

STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.04


# ---- MVTec AD Dataset ----
class MVTecADDataset(Dataset):
    """Load MVTec AD images from the standard directory structure.

    Expected structure:
        mvtec_anomaly_detection/
            bottle/
                train/good/*.png
                test/good/*.png
                test/broken_large/*.png
                ...
            cable/
                ...
    """

    def __init__(self, root_dir, categories=None, split="train",
                 defect_type="good", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []       # category index
        self.categories = categories or MVTEC_CATEGORIES
        self.defect_types = []  # "good" or specific defect name
        self.category_names = []

        for cat_idx, category in enumerate(self.categories):
            cat_dir = self.root_dir / category / split

            if defect_type == "all":
                # Load all defect types (for test set)
                if not cat_dir.exists():
                    continue
                for dtype_dir in sorted(cat_dir.iterdir()):
                    if not dtype_dir.is_dir():
                        continue
                    for img_path in sorted(dtype_dir.glob("*.png")):
                        self.samples.append(img_path)
                        self.labels.append(cat_idx)
                        self.defect_types.append(dtype_dir.name)
                        self.category_names.append(category)
            else:
                # Load specific defect type (e.g., "good" for training)
                type_dir = cat_dir / defect_type
                if not type_dir.exists():
                    continue
                for img_path in sorted(type_dir.glob("*.png")):
                    self.samples.append(img_path)
                    self.labels.append(cat_idx)
                    self.defect_types.append(defect_type)
                    self.category_names.append(category)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class MVTecADFromHuggingFace(Dataset):
    """Load MVTec AD from HuggingFace datasets library."""

    def __init__(self, split="train", defect_type="good", transform=None,
                 categories=None):
        from datasets import load_dataset

        self.transform = transform
        self.categories = categories or MVTEC_CATEGORIES
        self.samples = []
        self.labels = []
        self.defect_types = []
        self.category_names = []

        print(f"Loading MVTec AD from HuggingFace ({split}/{defect_type})...")

        for cat_idx, category in enumerate(self.categories):
            try:
                ds = load_dataset(
                    "mvtec-ad", category,
                    split=split,
                    trust_remote_code=True,
                )
            except Exception:
                try:
                    ds = load_dataset(
                        "alexriedel/MVTec-AD", category,
                        split=split,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    print(f"  Warning: Could not load {category}: {e}")
                    continue

            for item in ds:
                dtype = item.get("label", "good")
                if isinstance(dtype, int):
                    dtype = "good" if dtype == 0 else "defect"

                if defect_type == "all" or dtype == defect_type:
                    img = item["image"]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    self.samples.append(img.convert("RGB"))
                    self.labels.append(cat_idx)
                    self.defect_types.append(dtype)
                    self.category_names.append(category)

            print(f"  {category}: loaded {sum(1 for c in self.category_names if c == category)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_mvtec_data(data_root, transform_train, transform_eval):
    """Load MVTec AD data, trying local directory first, then HuggingFace."""
    data_path = Path(data_root) / "mvtec_anomaly_detection"

    if data_path.exists():
        print(f"Loading MVTec AD from local directory: {data_path}")
        train_dataset = MVTecADDataset(
            data_path, split="train", defect_type="good",
            transform=transform_train,
        )
        test_dataset_normal = MVTecADDataset(
            data_path, split="test", defect_type="good",
            transform=transform_eval,
        )
        test_dataset_all = MVTecADDataset(
            data_path, split="test", defect_type="all",
            transform=transform_eval,
        )
    else:
        print("Local MVTec AD not found, loading from HuggingFace...")
        train_dataset = MVTecADFromHuggingFace(
            split="train", defect_type="good",
            transform=transform_train,
        )
        test_dataset_normal = MVTecADFromHuggingFace(
            split="test", defect_type="good",
            transform=transform_eval,
        )
        test_dataset_all = MVTecADFromHuggingFace(
            split="test", defect_type="all",
            transform=transform_eval,
        )

    print(f"Training samples (normal only): {len(train_dataset)}")
    print(f"Test samples (normal): {len(test_dataset_normal)}")
    print(f"Test samples (all): {len(test_dataset_all)}")

    return train_dataset, test_dataset_normal, test_dataset_all


# ---- Training ----
def train_salt_mvtec(train_loader, device, epochs_s1=None, epochs_s2=None):
    """Run full SALT pipeline on MVTec AD, logging per-step losses."""
    epochs_s1 = epochs_s1 or STAGE1_EPOCHS
    epochs_s2 = epochs_s2 or STAGE2_EPOCHS

    num_patches = NUM_PATCHES
    grid_size = GRID_SIZE

    # ---- Stage 1: MAE Teacher ----
    print(f"\n{'='*60}")
    print(f"[Stage 1] Training MAE teacher ({epochs_s1} epochs)...")
    print(f"{'='*60}")

    teacher = build_encoder(ENC_CONFIG).to(device)
    decoder = build_mae_decoder(DEC_CONFIG, num_patches).to(device)
    pixel_loss_fn = PixelReconstructionLoss(patch_size=PATCH_SIZE, norm_pix=True)

    opt1 = AdamW(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    sched1 = CosineAnnealingLR(opt1, T_max=epochs_s1, eta_min=1e-6)

    trainer1 = SALTStage1Trainer(teacher, decoder, pixel_loss_fn, opt1, sched1, device)

    stage1_steps = []
    global_step = 0
    t0 = time.time()

    for epoch in range(epochs_s1):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            loss = trainer1.train_step(images, grid_size)
            global_step += 1
            epoch_loss += loss
            num_batches += 1

            if global_step % 5 == 0:
                stage1_steps.append({"step": global_step, "loss": round(loss, 4)})

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{epochs_s1} | Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.0f}s | Steps: {global_step}")
        sched1.step()

    print(f"  Stage 1 complete. Final loss: {avg_loss:.4f} | Total time: {time.time()-t0:.0f}s")

    # ---- Stage 2: Frozen Teacher JEPA ----
    print(f"\n{'='*60}")
    print(f"[Stage 2] Training student with frozen teacher ({epochs_s2} epochs)...")
    print(f"{'='*60}")

    frozen_teacher = copy.deepcopy(teacher)
    frozen_teacher.eval()
    for p in frozen_teacher.parameters():
        p.requires_grad = False

    student = build_encoder(ENC_CONFIG).to(device)
    predictor = build_predictor(PRED_CONFIG, num_patches).to(device)

    opt2 = AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    sched2 = CosineAnnealingLR(opt2, T_max=epochs_s2, eta_min=1e-6)

    trainer2 = SALTStage2Trainer(frozen_teacher, student, predictor, opt2, sched2, device)

    stage2_steps = []
    global_step = 0
    t0 = time.time()

    for epoch in range(epochs_s2):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            loss = trainer2.train_step(images, grid_size)
            global_step += 1
            epoch_loss += loss
            num_batches += 1

            if global_step % 5 == 0:
                stage2_steps.append({"step": global_step, "loss": round(loss, 4)})

        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{epochs_s2} | Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.0f}s | Steps: {global_step}")
        sched2.step()

    print(f"  Stage 2 complete. Final loss: {avg_loss:.4f} | Total time: {time.time()-t0:.0f}s")

    training_metrics = {
        "stage1": stage1_steps,
        "stage2": stage2_steps,
        "config": {
            "img_size": IMG_SIZE,
            "patch_size": PATCH_SIZE,
            "embed_dim": ENC_CONFIG["embed_dim"],
            "depth": ENC_CONFIG["depth"],
            "grid_size": GRID_SIZE,
            "num_patches": NUM_PATCHES,
            "stage1_epochs": epochs_s1,
            "stage2_epochs": epochs_s2,
            "batch_size": BATCH_SIZE,
        },
    }

    return student, training_metrics


# ---- Generate reference embeddings for anomaly detection ----
def generate_reference_embeddings(student, dataset, device, category_names=None):
    """
    Generate per-category reference embeddings from normal training images.

    For each category, stores:
    - Mean global embedding (for image-level anomaly score)
    - Per-patch mean embeddings (14×14 grid, for anomaly heatmap)

    These are compared against at inference time to detect anomalies.
    """
    student.eval()
    print("\n[Reference Embeddings] Generating per-category normal references...")

    # Group by category
    cat_embeddings = {cat: [] for cat in MVTEC_CATEGORIES}
    cat_patch_embeddings = {cat: [] for cat in MVTEC_CATEGORIES}

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Get patch-level embeddings: (B, 196, 192)
            patch_embs = student(images)
            # Global embedding: mean pool across patches
            global_embs = patch_embs.mean(dim=1)  # (B, 192)
            global_embs = F.normalize(global_embs, dim=1)

            for i in range(images.shape[0]):
                cat_name = MVTEC_CATEGORIES[labels[i].item()]
                cat_embeddings[cat_name].append(global_embs[i].cpu().numpy())
                # Normalize each patch embedding for cosine similarity
                patch_norm = F.normalize(patch_embs[i], dim=1)  # (196, 192)
                cat_patch_embeddings[cat_name].append(patch_norm.cpu().numpy())

    # Compute per-category statistics
    reference_data = {
        "categories": MVTEC_CATEGORIES,
        "embed_dim": ENC_CONFIG["embed_dim"],
        "num_patches": NUM_PATCHES,
        "grid_size": GRID_SIZE,
        "references": {},
    }

    for cat in MVTEC_CATEGORIES:
        if not cat_embeddings[cat]:
            print(f"  {cat}: no samples, skipping")
            continue

        global_embs = np.array(cat_embeddings[cat])
        patch_embs = np.array(cat_patch_embeddings[cat])

        # Mean global embedding for image-level scoring
        mean_global = global_embs.mean(axis=0)
        mean_global = mean_global / (np.linalg.norm(mean_global) + 1e-8)

        # Mean per-patch embeddings for heatmap (196 patches × 192 dim)
        mean_patches = patch_embs.mean(axis=0)  # (196, 192)
        # Normalize each patch
        patch_norms = np.linalg.norm(mean_patches, axis=1, keepdims=True) + 1e-8
        mean_patches = mean_patches / patch_norms

        reference_data["references"][cat] = {
            "global_embedding": mean_global.tolist(),
            "patch_embeddings": mean_patches.tolist(),
            "num_samples": len(cat_embeddings[cat]),
        }
        print(f"  {cat}: {len(cat_embeddings[cat])} reference images")

    return reference_data


# ---- Generate sample images for the demo ----
def generate_sample_images(data_root, num_per_category=6):
    """Extract sample images (normal + defective) as base64 for the demo grid."""
    data_path = Path(data_root) / "mvtec_anomaly_detection"
    samples = {}

    for category in MVTEC_CATEGORIES:
        cat_samples = []

        # Try local directory first
        test_dir = data_path / category / "test"
        if test_dir.exists():
            # Get normal samples
            good_dir = test_dir / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png"))[:2]:
                    cat_samples.append(_encode_sample(img_path, category, "good"))

            # Get defective samples (from various defect types)
            for dtype_dir in sorted(test_dir.iterdir()):
                if dtype_dir.name == "good" or not dtype_dir.is_dir():
                    continue
                for img_path in sorted(dtype_dir.glob("*.png"))[:1]:
                    cat_samples.append(_encode_sample(img_path, category, dtype_dir.name))
                if len(cat_samples) >= num_per_category:
                    break
        else:
            # HuggingFace fallback — load test split
            try:
                from datasets import load_dataset
                ds = load_dataset("mvtec-ad", category, split="test",
                                  trust_remote_code=True)
                count = 0
                for item in ds:
                    if count >= num_per_category:
                        break
                    img = item["image"]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                    dtype = item.get("label", "unknown")
                    if isinstance(dtype, int):
                        dtype = "good" if dtype == 0 else "defect"
                    cat_samples.append({
                        "base64": b64,
                        "category": category,
                        "defect_type": str(dtype),
                        "is_defective": dtype != "good",
                    })
                    count += 1
            except Exception as e:
                print(f"  Warning: Could not load samples for {category}: {e}")

        samples[category] = cat_samples[:num_per_category]
        print(f"  {category}: {len(samples[category])} sample images")

    return samples


def _encode_sample(img_path, category, defect_type):
    """Load, resize, and base64-encode a single sample image."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "base64": b64,
        "category": category,
        "defect_type": defect_type,
        "is_defective": defect_type != "good",
    }


# ---- ONNX Export ----
def export_model(student, output_dir, device):
    """Save checkpoint and export to ONNX."""
    print("\n[Export] Saving checkpoint and exporting ONNX...")

    ckpt_path = "/tmp/salt_inspector.pt"
    torch.save(student.state_dict(), ckpt_path)

    from scripts.export_onnx import export_to_onnx
    onnx_path = str(output_dir / "models" / "salt-inspector.onnx")

    export_to_onnx(
        checkpoint_path=ckpt_path,
        output_path=onnx_path,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=ENC_CONFIG["embed_dim"],
        depth=ENC_CONFIG["depth"],
        num_heads=ENC_CONFIG["num_heads"],
        fp16=False,  # Keep fp32 for browser WASM compatibility
    )

    return onnx_path


# ---- Main ----
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate MVTec AD web demo data")
    parser.add_argument("--output-dir", type=str, default="site",
                        help="Output directory for web assets")
    parser.add_argument("--data-root", type=str, default="/tmp/mvtec_data",
                        help="Root directory for MVTec AD data")
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("SALT Industrial Inspector — Data Generator")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: ViT {ENC_CONFIG['embed_dim']}d / {ENC_CONFIG['depth']}L / "
          f"{IMG_SIZE}px / patch {PATCH_SIZE}")
    print(f"Training: Stage 1 = {args.stage1_epochs}ep, Stage 2 = {args.stage2_epochs}ep")
    print(f"Output: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    # ---- 1. Load MVTec AD ----
    print("\n[1/5] Loading MVTec AD dataset...")
    train_transform = get_image_transforms(img_size=IMG_SIZE, is_train=True)
    eval_transform = get_image_transforms(img_size=IMG_SIZE, is_train=False)

    train_dataset, test_normal, test_all = load_mvtec_data(
        args.data_root, train_transform, eval_transform,
    )

    # Also load train set with eval transforms for reference embedding generation
    train_eval_dataset, _, _ = load_mvtec_data(
        args.data_root, eval_transform, eval_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
    )

    # ---- 2. Train SALT ----
    print("\n[2/5] Training SALT pipeline...")
    t_start = time.time()
    student, training_metrics = train_salt_mvtec(
        train_loader, device,
        epochs_s1=args.stage1_epochs,
        epochs_s2=args.stage2_epochs,
    )
    training_metrics["total_time_seconds"] = round(time.time() - t_start)
    print(f"\nTotal training time: {training_metrics['total_time_seconds']}s")

    with open(output_dir / "data" / "training-metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    print("  Saved training metrics")

    # ---- 3. Export ONNX ----
    print("\n[3/5] Exporting ONNX model...")
    onnx_path = export_model(student, output_dir, device)

    # ---- 4. Generate reference embeddings ----
    print("\n[4/5] Generating reference embeddings...")
    ref_data = generate_reference_embeddings(
        student, train_eval_dataset, device,
    )
    with open(output_dir / "data" / "reference-embeddings.json", "w") as f:
        json.dump(ref_data, f)
    ref_size = Path(output_dir / "data" / "reference-embeddings.json").stat().st_size
    print(f"  Saved reference embeddings ({ref_size / 1024:.0f} KB)")

    # ---- 5. Generate sample images ----
    print("\n[5/5] Generating sample images for demo grid...")
    samples = generate_sample_images(args.data_root)
    with open(output_dir / "data" / "mvtec-samples.json", "w") as f:
        json.dump(samples, f)
    samples_size = Path(output_dir / "data" / "mvtec-samples.json").stat().st_size
    print(f"  Saved sample images ({samples_size / (1024*1024):.1f} MB)")

    # ---- Done ----
    print("\n" + "=" * 60)
    print("All data generated successfully!")
    print(f"Output: {output_dir}")
    print(f"  models/salt-inspector.onnx")
    print(f"  data/training-metrics.json")
    print(f"  data/reference-embeddings.json")
    print(f"  data/mvtec-samples.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
