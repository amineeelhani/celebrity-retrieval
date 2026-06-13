# Celebrity Retrieval Across Domains

**Introduction to Machine Learning — Competition Project**
**University of Trento — M.Sc. Data Science — Spring 2026**
**Challenge date: 21 May 2026 · Group: Trade-Off**

Cross-domain celebrity image retrieval: given a real photograph of a celebrity, retrieve the most likely matching images of the same identity from a gallery of **synthetically generated** portraits. The central difficulty is the **domain shift** between natural photographs and generated faces.

Our best competition-day result was **CLIP ViT-L/14 with horizontal-flip Test-Time Augmentation (669.0 / 1000)**, a 7.4× improvement over the ResNet50 baseline using only training-free techniques. After the competition we found and fixed three bugs that had silently disabled our ArcFace fine-tuning pipeline; the corrected pipeline reaches **752.7 / 1000** (+83.7).

---

## Problem Description

Each **query** is a natural photograph of a celebrity; the **gallery** contains synthetically generated images (diffusion/GAN-style portraits) of the same set of identities. For every query the system returns a ranked list of **10 gallery images**, ideally placing same-identity images as close to the top as possible.

The score is computed out of **1000 points**:

| Metric | Description | Points |
|--------|-------------|--------|
| Top-1  | The correct identity is the first retrieved image | 600 |
| Top-5  | The correct identity appears in the first 5 results | 300 |
| Top-10 | The correct identity appears in the first 10 results | 100 |

Because Top-1 carries most of the points, the quality of the embedding space — how tightly same-identity images cluster versus other identities — is the main driver of the final score.

This is **not** a classification task: the system never predicts an identity label. Queries are matched to gallery images purely by similarity in an embedding space, so what matters is the geometry of that space across the two domains. Our solution therefore builds on a strong pretrained joint-embedding model (CLIP), ranks by cosine similarity, and optionally sharpens the space with a metric-learning loss (ArcFace).

---

## Repository Structure

```
celebrity-retrieval/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── baseline/
│   └── baseline_resnet50.py              # Professor's baseline (ImageNet ResNet50)
│
├── models/
│   ├── clip_retrieval.py                 # CLIP ViT-B/32 zero-shot retrieval
│   ├── dinov2_retrieval.py               # DINOv2 ViT-B/14 zero-shot retrieval
│   ├── clip_ensemble_retrieval.py        # Ensemble: ViT-B/32 + ViT-L/14 (concatenated)
│   ├── clip_tta_retrieval.py             # CLIP ViT-L/14 + flip TTA — BEST COMPETITION RESULT
│   ├── clip_finetuned_retrieval.py       # Retrieval with a fine-tuned CLIP ViT-B/32
│   ├── clip_vitl14_finetuned_retrieval.py# Retrieval with the fine-tuned CLIP ViT-L/14
│   ├── finetune_clip_crossentropy.py     # Fine-tuning with CrossEntropy (ViT-B/32)
│   ├── finetune_clip_arcface.py          # Fine-tuning with ArcFace (ViT-B/32)
│   └── finetune_clip_vit14_arcface.py    # Fine-tuning with ArcFace (ViT-L/14) — FINAL / FIXED
│
├── utils/
│   ├── submit.py                         # submit() to the eval server + evaluate_local()
│   ├── face_detection.py                 # MTCNN wrapper (available, unused in final pipeline)
│   └── preprocess_faces.py               # Optional MTCNN face-cropping of a dataset
│
├── experiments/
│   └── results.md                        # Experiment log template
│
└── data/                                 # NOT tracked by git — local or VM only
    ├── train/                            # 6000 images in 1250 identity folders
    ├── query/                            # Real celebrity photos (test set)
    └── gallery/                          # Synthetic celebrity images (test set)
```

> Always run scripts from the **root of the repository** (the fine-tuning utilities resolve `utils/` relative to the repo root).

---

## Setup

```bash
git clone https://github.com/amineeelhani/celebrity-retrieval.git
cd celebrity-retrieval

# CUDA 12.1 wheels are required for the pinned torch build
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` pins the key dependencies:

```
torch==2.2.0+cu121
torchvision==0.17.0+cu121
numpy==1.26.4
Pillow
requests
timm
facenet-pytorch
git+https://github.com/openai/CLIP.git
```

### Data

The dataset is **not** tracked by git. Place it under `data/` (or point each script's data path at your local/VM directory):

- `data/train/` — provided labeled training set: **6000 images in 1250 identity folders**, very unevenly composed of **250 celebrity identities with 20 natural photos each** plus **1000 single-image synthetic identities** (SFHQ generated faces) acting as distractor classes. This is the only labeled data available and the target for fine-tuning.
- `data/query/` — real celebrity photographs (test queries).
- `data/gallery/` — synthetic celebrity images (test gallery).

Optionally, **VGGFace2** (112×112, Kaggle: 8631 identities, ~3.3M images) is used as an intermediate fine-tuning stage for the ViT-B/32 runs.

---

## Quick Start

Run the best competition configuration:

```bash
python models/clip_tta_retrieval.py
```

Each retrieval script embeds the query and gallery images, L2-normalizes them, ranks gallery images by cosine similarity, and returns the top-10 per query. Results are either submitted to the evaluation server or scored locally:

```python
from utils.submit import submit, evaluate_local

# submit to the server
submit(results, groupname="trade-off", url="http://<server>/retrieval/")

# or score locally against known ground truth
evaluate_local(results, ground_truth)   # prints Top-1/5/10 and the /1000 score
```

The submission format is a dict mapping each query filename to exactly 10 gallery filenames ordered by similarity:

```python
results = {
    "query_image_1.jpg": ["gallery_7.jpg", "gallery_2.jpg", ...],  # 10 images, ranked
    ...
}
```

---

## Retrieval Pipeline

Every configuration is evaluated through the **same** pipeline, so the comparison is clean:

1. Embed query and gallery images with the chosen backbone.
2. L2-normalize all embeddings.
3. Compute cosine similarity between every query and every gallery embedding.
4. For each query, retrieve the 10 highest-similarity gallery images.
5. Submit to the server or score locally with `evaluate_local`.

For the fine-tuned models, only the fine-tuned **visual encoder** is used at inference; the classification head is discarded.

---

## Results

| Model | Score / 1000 | Notes |
|-------|-------------:|-------|
| ResNet50 baseline | 90.6 | ImageNet-pretrained CNN encoder |
| DINOv2 ViT-B/14 zero-shot | 126.3 | Self-supervised transformer |
| CLIP ViT-B/32 zero-shot | 482.3 | Contrastive pretraining |
| Ensemble ViT-B/32 + ViT-L/14 | 619.3 | Concatenated, normalized embeddings |
| CLIP ViT-L/14@336px zero-shot | 659.3 | Higher-resolution input variant |
| CLIP ViT-L/14 zero-shot | 666.3 | Larger model than ViT-B/32 |
| **CLIP ViT-L/14 + TTA (flip)** | **669.0** | **Best score during the competition** |
| CLIP ViT-L/14 fine-tuned (ArcFace, fixed) + TTA | **752.7** | **Post-competition fix, +83.7** |

Because each row changes one design choice relative to another, the table doubles as an ablation. Three observations stand out:

1. **CLIP dominates ResNet50 and DINOv2 by a large margin** (482.3 vs 90.6 / 126.3). CLIP's diverse pretraining distribution transfers far better across the natural/synthetic domain gap than ImageNet-supervised or self-supervised features trained mostly on natural images.
2. **Model scale matters more than fancy combination strategies.** Switching ViT-B/32 → ViT-L/14 (+184.0) gave a far larger gain than concatenating both backbones into an ensemble — which actually scored *worse* than ViT-L/14 alone, as the smaller model's weaker embedding dilutes the larger one once re-normalized.
3. **Test-time augmentation is a small, free win.** Averaging the embedding of an image and its horizontal flip added +2.7 points at no training cost.

---

## Test-Time Augmentation (TTA)

TTA improves retrieval at inference with no extra training. Instead of embedding a single version of each image, we embed multiple augmented versions and average them. We use a **horizontal flip**:

```python
feat      = model.encode_image(inputs).float()
feat_flip = model.encode_image(inputs_flip).float()   # horizontally flipped
feat_avg  = (feat + feat_flip) / 2                     # averaged embedding
```

A celebrity photo may show the face looking left or right, and CLIP can assign slightly different embeddings depending on orientation. Averaging the original and flipped embeddings yields a more stable, orientation-robust representation. (A ±10° rotation variant was also tried but only marginally better than flip-only, so it was not used for the final submission.)

---

## ArcFace Loss

The ArcFace head (`ArcFaceLinear`, implemented from scratch) replaces the standard linear classifier: it L2-normalizes both the feature vector and the class weights so each logit depends only on the angle θⱼ between the feature and the j-th class center. ArcFace then adds an angular margin *m* to the ground-truth angle before scaling by *s*:

```
CrossEntropy maximizes:  cos(θ)
ArcFace maximizes:       cos(θ + m)     with m = 0.5 rad
```

The margin pushes same-identity embeddings closer to their class center and increases angular separation between identities — directly improving cosine-similarity retrieval, which is exactly the metric used at inference.

---

## Fine-tuning Pipeline

For the ViT-B/32 configurations, fine-tuning runs in two stages: first the visual encoder is pretrained on **VGGFace2** (all CLIP layers frozen except the last 4 transformer blocks, with a new classification head — linear for CrossEntropy, ArcFace otherwise), then the checkpoint is adapted to the professor's competition data (1250 identities), with the head resized accordingly. The post-competition **ViT-L/14** run skips the VGGFace2 stage and fine-tunes directly on `data/train/`.

Training augmentations: random horizontal flip, random rotation (±15°), color jitter (brightness/contrast/saturation, plus hue for ViT-B/32), occasional grayscale, and Gaussian blur (ViT-B/32). Each competition run uses an 80/20 split → **4800 train / 1200 validation** across 1250 identities, batch size 32, with the best-validation-epoch checkpoint retained.

| Stage | Optimizer | Head LR | CLIP LR (last 4) | Epochs |
|-------|-----------|--------:|-----------------:|-------:|
| VGGFace2 (initial) | Adam | 1e-3 | 1e-5 | 10 |
| Competition (initial, **buggy**) | Adam | 1e-3 | (frozen) | 10 |
| Competition (post-hoc **fixed**, ViT-L/14 ArcFace) | Adam | 1e-4 | 1e-6 | 10 |

---

## Why Fine-tuning Failed on Competition Day

During the competition, **all** fine-tuned configurations scored identically to CLIP ViT-B/32 zero-shot (482.3) — i.e. fine-tuning had *zero effect*. A post-hoc analysis revealed **three compounding bugs**, each on its own sufficient to prevent any learning.

**Bug 1 — Frozen layers, nothing was ever saved.** The script froze every parameter and only unfroze the last 4 transformer blocks inside a branch guarded by `TRAINING_MODE == "vggface2"`. The competition runs used `TRAINING_MODE = "competition"`, so the backbone stayed fully frozen. Since `torch.save(model.state_dict(), ...)` stores the *entire* CLIP backbone, the saved weights were **bit-for-bit identical to stock CLIP** — verified: **302/302 layers unchanged**.

**Bug 2 — `torch.no_grad()` blocked gradient flow.** Even after unfreezing, the training loop computed `features = model.encode_image(images)` inside a `torch.no_grad()` context, detaching the computation graph. The unfrozen blocks received no gradient and never updated, despite `requires_grad = True`.

**Bug 3 — float16 internals produced NaN gradients.** Once gradients finally flowed, CLIP's internal float16 ops overflowed the float16 range (max ≈ 65504) and the gradient norm collapsed to `nan` from the second batch onward:

```
Grad norm: 0.252441   <- first batch OK
Grad norm: nan        <- batch 2 onward
Epoch 1/10 | Train Loss: nan | Train Acc: 0.0% | Val Acc: 0.3%
```

An additional stability issue: `num_workers=4` occasionally **deadlocked** the DataLoader on the Azure VM, hanging for hours with no output.

| # | Problem | Fix | Severity |
|---|---------|-----|----------|
| 1 | All CLIP layers frozen → saved checkpoint = stock CLIP | Always unfreeze last 4 transformer blocks | Critical |
| 2 | `torch.no_grad()` around `encode_image` in training | Remove it from the training forward pass | Critical |
| 3a | float16 ops → NaN gradients | `model.float()` (train in float32) | Critical |
| 3b | Unbounded gradient norms | `clip_grad_norm_(max_norm=1.0)` | Critical |
| 3c | Learning rate too high once gradients flowed | head 1e-3→1e-4, CLIP 1e-5→1e-6 | Important |
| 3d | ArcFace logits too large/unstable | scale *s*: 64 → 16 | Important |
| 4 | DataLoader deadlock under load | `num_workers`: 4 → 0 | Stability |

---

## Post-Competition Fix — CLIP ViT-L/14 ArcFace

**File:** `models/finetune_clip_vit14_arcface.py` · **Score:** 752.7 / 1000 (+83.7 over the competition best)

After applying all fixes, we re-verified that the backbone weights actually changed (**0/446 layers identical to stock CLIP**) and re-trained for 10 epochs. Validation accuracy on the full dataset (4800 train / 1200 val, 1250 identities) climbed monotonically and gradient norms stayed stable (≈0.23–0.24, no NaNs):

| Epoch | 1 | 3 | 5 | 8 | 10 |
|-------|---:|---:|---:|---:|---:|
| Val Acc | 2.2% | 25.3% | 43.8% | 54.6% | 58.2% |

Re-running the retrieval pipeline with the fine-tuned encoder (keeping the same flip TTA, `models/clip_vitl14_finetuned_retrieval.py`) and submitting to the evaluation server — which stayed reachable after the deadline — gives **752.7 / 1000**. Same server, test set, scoring function and pipeline (only the encoder weights differ), so the +83.7 gain is attributable entirely to fine-tuning.

> **Note:** the 752.7 result is **post-hoc** (produced 22/05/2026, the day after the deadline) and is **not part of the official ranking**. The group's official competition score is **669.0 / 1000** (CLIP ViT-L/14 + TTA). The post-hoc number is reported for completeness, to show what the approach was capable of once the bugs were fixed.

---

## Qualitative Analysis

Since the official test labels are not available locally, we built a labeled retrieval split from the training data (60 natural queries against a 1000-image gallery: 5 held-out images for each of 120 celebrity identities, plus 400 SFHQ synthetic distractors). CLIP ViT-L/14 zero-shot reaches **Top-1 65.0%, Top-5 80.0%, Top-10 81.7%** on this split.

The failure cases are instructive: dark stage lighting causes scene-level attributes to dominate over identity, and a query in theatrical white face paint retrieves *only* synthetic faces — its smooth, uniform skin texture lands closer to the synthetic-face cluster than to natural photos of the same person. This is direct evidence that CLIP's embedding space encodes the natural/synthetic texture gap that defines the competition, and explains why ArcFace fine-tuning (which pulls same-identity embeddings together regardless of domain) helps once functional.

> This split pairs natural queries with *natural* same-identity gallery images, so its absolute numbers are not directly comparable to the server scores, which require the harder natural→synthetic matching.

---

## Hardware

- **Local development:** Windows PC.
- **Training:** Azure VM — Tesla V100 16GB, CUDA 12.1, Ubuntu 20.04.
- **Stack:** PyTorch 2.2.0+cu121, torchvision 0.17.0, OpenAI CLIP, facenet-pytorch (MTCNN, available but unused in the final pipeline), Pillow.

---

## Key Takeaway

The single largest factor in our results is **backbone choice** (ImageNet CNN / self-supervised ViT → CLIP), consistent with the hypothesis that pretraining-data diversity drives robustness to the domain gap. On top of a strong CLIP backbone, training-free tricks (TTA) give small reliable gains. The biggest lesson from the fine-tuning story: the largest single improvement of the project came not from a new idea but from **fixing implementation bugs** that left the backbone weights unchanged while the training loop still ran and printed plausible-looking numbers. Always verify that a fine-tuning pipeline is *actually updating the intended weights* (e.g. a before/after weight-diff check) before trusting its training curves.

---

## Team — Trade-Off

- Mohamed Amine El Hani
- Angelica Fiorio
- Francesco Leoni

Most of the project was developed jointly (retrieval pipeline, zero-shot, ensemble and TTA experiments, and this report). The fine-tuning experiments were divided: Francesco Leoni — ArcFace ViT-B/32; Mohamed Amine El Hani — CrossEntropy ViT-B/32; Angelica Fiorio — post-competition ViT-L/14 ArcFace. All members shared management of the Azure training infrastructure.