# Celebrity Retrieval Across Domains
**Introduction to Machine Learning — Competition Project**  
Challenge date: May 21, 2025

---

## Problem Description

Given a set of real celebrity photographs (**query**) and a set of synthetic celebrity images (**gallery**), the system must return for each query the 10 most similar gallery images, ranked by similarity.

The maximum score is **1000 points**:

| Metric  | Description                                  | Points |
|---------|----------------------------------------------|--------|
| Top-1   | The first retrieved image is correct         | 600    |
| Top-5   | At least 1 correct image in the first 5      | 300    |
| Top-10  | At least 1 correct image in the first 10     | 100    |

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
│   └── baseline_resnet50.py      # professor's template
│
├── models/
│   └── (fine-tuned model checkpoints)
│
├── utils/
│   └── submit.py                 # submit function + local evaluator
│
├── experiments/
│   └── results.md                # experiment log with scores
│
└── data/                         # NOT tracked by git — local or VM only
    ├── train/                    # ~5000 labeled images for training
    │   ├── identity1/
    │   │   ├── aaa.jpg
    │   │   └── bbb.jpg
    │   └── identity2/
    │       ├── aaa.jpg
    │       └── bbb.jpg
    ├── query/                    # real photos (test set)
    └── gallery/                  # synthetic images (test set)
```

---

## Setup

```bash
git clone https://github.com/amineeelhani/celebrity-retrieval.git
cd celebrity-retrieval
pip install -r requirements.txt
```

---

## Running the Baseline

```bash
python baseline/baseline_resnet50.py
```

Edit the `data_folder` path in the script to point to your local or VM data directory.

---

## Approach

### Baseline — ResNet50 (ImageNet)
Feature extraction with a pretrained ResNet50, cosine similarity for ranking.

### Target — CLIP
CLIP (ViT-B/32) was trained on diverse image-text pairs across many visual domains, making it naturally robust to the domain shift between real photos and synthetic images.

---

## Submission Format

```python
results = {
    "query_image_1.jpg": [
        "gallery_image_7.jpg",
        "gallery_image_2.jpg",
        ...   # exactly 10 gallery images, ordered by similarity
    ],
    ...
}
submit(results, groupname="trade-off", url="http://...")
```

---

## Team

- Mohamed Amine El Hani
- Angelica Fiorio
- Francesco Leoni
