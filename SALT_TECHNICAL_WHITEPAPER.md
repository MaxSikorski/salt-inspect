# SALT Inspector: Self-Supervised Anomaly Detection for Industrial Visual Inspection

**A Technical Overview for Engineering Leaders**

*Sikorski AI Research Lab — March 2026*
*max@sikorski.ai | salt-inspect.com | github.com/MaxSikorski/salt-inspect*

---

## Abstract

**SALT Inspector delivers custom anomaly detection results in 48 hours.** Customers send photos of good parts; we return a working inspector accessible from any browser. No defect images needed. No install required.

Under the hood, SALT Inspector is built on Apple's SALT (Static-teacher Asymmetric Latent Training) architecture, published at ICLR 2025. Unlike traditional machine vision systems that require labeled defect data and proprietary hardware, SALT Inspector trains exclusively on images of normal parts and deploys to any device via ONNX Runtime — including web browsers. This whitepaper describes the architecture, training pipeline, deployment options, and benchmark results, providing technical decision-makers with the information needed to evaluate SALT Inspector for their inspection needs.

---

## 1. The Problem with Current Approaches

### 1.0 The Knowing-Doing Gap

Every manufacturer we talk to knows they need AI inspection. They've seen the Cognex booth at Automate. They've read the trade publications. They've been meaning to automate visual inspection for years. The problem isn't awareness — it's activation. Every current approach requires defect data they don't have, proprietary hardware they can't justify, or 6-month integration timelines they can't commit to. The result: AI inspection stays on the backlog. SALT Inspector collapses this knowing-doing gap to 48 hours.

Industrial visual inspection has three dominant paradigms, each with significant limitations:

### 1.1 Rule-Based Machine Vision
Traditional systems (Cognex VisionPro, Keyence CV-X) use hand-engineered algorithms: edge detection, template matching, blob analysis, and color thresholding. These systems are fragile — any change in lighting, part orientation, or surface finish requires manual re-tuning by a vision engineer. They work well for simple, high-contrast defects on uniform surfaces but fail on complex textures, subtle anomalies, and natural part variation.

### 1.2 Supervised Deep Learning
Modern deep learning approaches (Cognex ViDi Red Tool, Landing AI) train convolutional neural networks on labeled datasets of defective and non-defective images. This achieves high accuracy but creates a data dependency: you need hundreds of labeled defect images per defect class. For new product launches, rare defect types, and custom parts, this data simply doesn't exist. The labeling process itself requires domain expertise and typically takes weeks.

### 1.3 Unsupervised Anomaly Detection (Pre-SALT)
Methods like PatchCore, PaDiM, and EfficientAD train on normal images only, learning a distribution of "normal" features. Anything outside this distribution is flagged as anomalous. This eliminates the labeled data requirement but introduces architectural complexity:

- **PatchCore** (2022): Stores a coreset of patch-level features from a pre-trained ImageNet backbone. Achieves 99.1% image-level AUROC on MVTec AD but requires large memory banks and depends on ImageNet pre-training.
- **EfficientAD** (2023): Uses student-teacher distillation with an autoencoder. Lightweight but still relies on ImageNet features and exponential moving average (EMA) for teacher updates — a source of training instability.

All of these methods share a common limitation: they depend on features learned from ImageNet (natural images of dogs, cars, landscapes), not from industrial parts. The feature space is misaligned with the inspection domain.

---

## 2. The SALT Approach

### 2.1 Overview

SALT (Static-teacher Asymmetric Latent Training) was published by Apple Research at ICLR 2025. It introduces a two-stage self-supervised training pipeline that eliminates the EMA entirely — the most complex and memory-intensive component of prior JEPA (Joint-Embedding Predictive Architecture) methods.

The key insight: **a frozen teacher trained via pixel reconstruction provides a more stable and efficient learning signal than an EMA-updated teacher.**

### 2.2 Stage 1: MAE Teacher (V-Pixel)

A Vision Transformer (ViT) is trained as a Masked Autoencoder (MAE):

1. Input: 224×224 RGB image, divided into 16×16 patches (196 patches total)
2. Masking: 75% of patches are randomly masked
3. Task: Reconstruct the masked patches at the pixel level
4. Loss: Mean Squared Error between predicted and original pixel values

After Stage 1, the encoder has learned rich spatial representations of the training domain — the visual structure of your specific manufactured parts, not generic ImageNet features.

**SALT Inspector config:**
- Encoder: ViT with embed_dim=192, depth=12, num_heads=3
- Patch size: 16×16
- Training: 200 epochs on normal manufacturing images
- Typical Stage 1 loss trajectory: 0.82 → 0.49 (converges around epoch 80-100)

### 2.3 Stage 2: JEPA Student (Frozen Teacher)

The Stage 1 encoder is frozen and becomes the teacher. A new student encoder is trained to match the teacher's latent representations:

1. Input: Same 224×224 images
2. Teacher: Frozen Stage 1 encoder produces target patch embeddings
3. Student: New encoder predicts the teacher's embeddings for masked patches
4. Loss: L1 loss between student predictions and teacher targets in latent space

**Why this works better than EMA:**
- No training instability from EMA momentum scheduling
- No memory overhead from maintaining a shadow copy of the model
- The "static teacher" provides a consistent learning target
- Apple showed this achieves better downstream performance with ~30% fewer FLOPs than V-JEPA 2

**The "weak teacher, strong student" principle:** The Stage 1 teacher doesn't need to be large or highly optimized. A smaller, simpler teacher can effectively train a larger, more powerful student. This means training costs are dominated by Stage 2 student training, not teacher quality.

### 2.4 Anomaly Detection Pipeline

After two-stage training, the student encoder produces a 192-dimensional embedding for each of the 196 patches in an input image.

**Reference building:**
1. Run all normal training images through the student encoder
2. Collect patch embeddings: shape [N_images × 196 × 192]
3. Compute per-patch mean embeddings as "normal reference": shape [196 × 192]

**Inference (anomaly detection):**
1. Encode the test image: [1 × 196 × 192]
2. Compute cosine similarity between each test patch and corresponding reference patch
3. Similarity < threshold → anomalous patch
4. Reshape similarity scores to 14×14 grid → anomaly heatmap
5. Overlay heatmap on original image for visualization

**Per-patch localization** is a key advantage: the system doesn't just say "this part is defective" — it shows *where* the defect is on a 14×14 spatial grid. Each cell covers a 16×16 pixel region of the original 224×224 image.

---

## 3. Architecture Specifications

| Component | Specification |
|-----------|---------------|
| Input resolution | 224 × 224 × 3 (RGB) |
| Patch size | 16 × 16 pixels |
| Patch grid | 14 × 14 (196 patches) |
| Encoder | Vision Transformer (ViT) |
| Embedding dimension | 192 |
| Transformer depth | 12 layers |
| Attention heads | 3 |
| Model parameters | ~11M |
| ONNX model size | ~18MB (fp32) |
| Inference output | [1, 196, 192] patch embeddings |
| Anomaly map resolution | 14 × 14 (upsampled to input resolution for display) |

---

## 4. Training Pipeline

### 4.1 Data Requirements

| Requirement | Detail |
|-------------|--------|
| Normal images | 100–500 minimum (more is better, diminishing returns above ~2,000) |
| Image quality | Standard production lighting, any camera (smartphone acceptable) |
| Defect images | Not required for training (optional for evaluation) |
| Labeling | None required |
| Image format | Any (JPEG, PNG, TIFF — resized to 224×224 during preprocessing) |
| Domain diversity | Capture natural variation: lighting, orientation, surface finish |

### 4.2 Compute Requirements

| Stage | Hardware | Time (3,629 images) |
|-------|----------|---------------------|
| Stage 1 (MAE) | NVIDIA A100 | ~90 min (200 epochs) |
| Stage 2 (JEPA) | NVIDIA A100 | ~90 min (200 epochs) |
| ONNX Export | CPU | <1 min |
| Reference Generation | GPU or CPU | ~5 min |
| **Total** | **A100** | **~3 hours** |

Cloud GPU cost: approximately $4–8 per training run on Google Colab, RunPod, or Lambda Labs.

### 4.3 Training Output

The pipeline produces four artifacts ready for deployment:

1. **ONNX model** (~18MB) — Student encoder exported for cross-platform inference
2. **Reference embeddings** (JSON/binary) — Per-patch mean embeddings from normal training set
3. **Training metrics** (JSON) — Stage 1/2 loss curves, timing data
4. **Sample anomaly maps** (PNG) — Visualizations for quality verification

---

## 5. Deployment Options

### 5.1 Browser (WebAssembly)

ONNX Runtime Web runs the model directly in the browser via WebAssembly (WASM). No server, no installation, no data upload.

- **Inference time:** <100ms on modern desktop browsers, ~200-500ms on mobile
- **Use case:** Demos, evaluations, low-volume manual inspection, mobile spot-checks
- **Privacy:** All computation runs client-side; images never leave the device
- **Compatibility:** Chrome, Firefox, Safari, Edge (any browser with WASM support)

### 5.2 Edge Devices

| Platform | Runtime | Typical Inference | Cost |
|----------|---------|-------------------|------|
| NVIDIA Jetson Orin Nano | ONNX Runtime + CUDA | ~10-20ms | ~$200 |
| Apple Neural Engine (M-series) | Core ML (converted from ONNX) | ~5-15ms | Built into Mac/iPhone |
| Intel OpenVINO | ONNX → OpenVINO IR | ~15-30ms | Existing x86 hardware |
| Raspberry Pi 5 | ONNX Runtime CPU | ~200-500ms | ~$80 |

### 5.3 Cloud API

For high-throughput applications, deploy the ONNX model behind a REST API (FastAPI, Flask) on any cloud provider. Batch processing of images from existing camera systems.

---

## 6. Benchmark Results

### 6.1 MVTec AD Training

Trained on the full MVTec AD dataset (3,629 normal training images, 15 categories):

**Stage 1 — MAE Teacher:**
- Initial loss: 0.82
- Final loss: 0.49
- Convergence: ~epoch 80-100, stable through epoch 200

**Stage 2 — JEPA Student:**
- Initial loss: 0.51
- Final loss: 0.43
- Convergence: ~epoch 100-150, stable through epoch 200

### 6.2 How SALT Compares to Published Methods

The MVTec AD benchmark is well-established with published AUROC scores:

| Method | Year | Image AUROC | Approach |
|--------|------|------------|----------|
| InvAD | 2024 | 99.7% | Inverse distillation |
| UniAD | 2024 | 99.5% | Unified framework |
| PatchCore | 2022 | 99.1% | Coreset + ImageNet features |
| EfficientAD | 2023 | 99.1% | Student-teacher + autoencoder |
| PaDiM | 2021 | 97.9% | Gaussian patch modeling |

**Note:** Direct AUROC comparison requires standardized evaluation protocols. Our implementation focuses on practical deployment characteristics (speed, size, data requirements, deployment flexibility) rather than purely maximizing benchmark scores. The SALT architecture is architecturally superior to PatchCore/PaDiM approaches because it learns domain-specific features (not ImageNet features) and doesn't require large memory banks.

### 6.3 Practical Advantages Over Benchmark Leaders

| Factor | SALT Inspector | PatchCore | EfficientAD |
|--------|---------------|-----------|-------------|
| Pre-trained backbone? | No (trains from scratch on your data) | Yes (ImageNet) | Yes (ImageNet) |
| Memory bank? | No (single reference embedding set) | Yes (large coreset) | No |
| Domain-specific features? | Yes (learned on your parts) | No (ImageNet features) | No (ImageNet features) |
| Model size | ~18MB | ~500MB+ (with coreset) | ~50MB |
| Browser deployment? | Yes (ONNX + WASM) | Difficult (large memory bank) | Possible but heavy |
| Training on custom data | Native (the whole point) | Possible but slower | Possible but slower |

---

## 7. Integration Architecture

### 7.1 Typical Production Deployment

```
Camera (GigE Vision / USB3)
    ↓
Edge Computer (Jetson / x86 / Mac)
    ↓ ONNX Runtime inference (~10-30ms)
Anomaly Score + Heatmap
    ↓
Decision Logic (pass/fail threshold)
    ↓
PLC Signal (accept/reject)  +  Dashboard (monitoring/logging)
```

### 7.2 Integration Points

- **Camera input:** Any camera supported by OpenCV (GigE Vision, USB3, MIPI CSI)
- **PLC communication:** Modbus TCP, OPC-UA, digital I/O via edge GPIO
- **MES/ERP:** REST API for logging inspection results to production databases
- **Dashboard:** Web-based monitoring of detection rates, false positive rates, throughput
- **Alerting:** Email/SMS/Slack notifications on anomaly rate spikes

---

## 8. Getting Started

### Option 1: Try the Browser Demo (Free, 60 seconds)
Visit [salt-inspect.com](https://salt-inspect.com) and upload any image. See per-patch anomaly heatmaps running entirely in your browser. This is a preview of what your custom results will look like.

### Option 2: Get Custom Results ($5,000, 48 hours)
Send us photos of your normal parts. We train a custom SALT model and deliver a browser-based inspector tuned to YOUR specific parts. You touch nothing — no software to install, no ML expertise needed. If the AI doesn't find defects, you don't pay.

### Option 3: Scale to Production ($100,000, 4–8 weeks)
Full production-grade deployment. Edge hardware, monitoring dashboard, operator training, MES/ERP integration. Hardened for 24/7 operation. Plus $5K/month ongoing retainer for monitoring and quarterly retraining.

### Contact
Max Sikorski — Sikorski AI Research Lab
Email: max@sikorski.ai
Web: salt-inspect.com
GitHub: github.com/MaxSikorski/salt-inspect

---

## References

1. Apple Research. "SALT: Static-teacher Asymmetric Latent Training." ICLR 2025. arXiv:2509.24317
2. Bergmann et al. "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR 2019.
3. He et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
4. Roth et al. "Towards Total Recall in Industrial Anomaly Detection." (PatchCore) CVPR 2022.
5. Batzner et al. "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies." WACV 2024.
6. Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." (I-JEPA) CVPR 2023.

---

*Sikorski AI Research Lab — March 2026*
*This document is provided for technical evaluation purposes. SALT Inspector is open source under the project license.*
