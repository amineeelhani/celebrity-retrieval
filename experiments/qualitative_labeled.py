"""
Labeled qualitative retrieval experiment for the report (Section 3.4).

data/train structure (verified):
  - 250 celebrity identities with 20 natural photographs each;
  - 1000 synthetic distractor identities (SFHQ) with 1 image each.

Split construction (seeded, reproducible):
  - sample N_CELEBS celebrity identities;
  - for each, hold out 1 image as a query candidate, put G_PER_ID of the
    remaining images in the gallery (correct matches always present);
  - add N_DISTRACTORS SFHQ synthetic faces to the gallery as distractors.

Runs CLIP ViT-L/14 zero-shot retrieval, computes Top-1/5/10 + competition
score on the split, and saves success / failure / ambiguous example rows as
report/figures/qualitative_labeled.png plus a JSON summary.
"""

import json
import os
import random

import torch
import clip
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "data", "train")
OUT_FIG = os.path.join(ROOT, "report", "figures", "qualitative_labeled.png")
OUT_JSON = os.path.join(ROOT, "experiments", "qualitative_labeled_results.json")

SEED = 42
N_CELEBS = 120        # celebrity identities sampled
N_QUERIES = 60        # of which this many get a query image
G_PER_ID = 5          # gallery images per sampled celebrity
N_DISTRACTORS = 400   # SFHQ synthetic distractor images in the gallery
BATCH = 16

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()

# ---------------------------------------------------------------- build split
all_ids = sorted(
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
)
celebs = [d for d in all_ids if not d.startswith("sfhq")]
sfhq = [d for d in all_ids if d.startswith("sfhq")]
print(f"Celebrities: {len(celebs)}  SFHQ distractor identities: {len(sfhq)}")

sampled = random.sample(celebs, N_CELEBS)

queries = []   # (identity, path)
gallery = []   # (identity, path)
for ident in sampled:
    folder = os.path.join(TRAIN_DIR, ident)
    imgs = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    queries.append((ident, imgs[0]))
    gallery.extend((ident, p) for p in imgs[1:1 + G_PER_ID])

for ident in random.sample(sfhq, N_DISTRACTORS):
    folder = os.path.join(TRAIN_DIR, ident)
    f = next(f for f in sorted(os.listdir(folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png")))
    gallery.append((ident, os.path.join(folder, f)))

random.shuffle(queries)
queries = queries[:N_QUERIES]
random.shuffle(gallery)
print(f"Queries: {len(queries)}  Gallery: {len(gallery)} "
      f"({N_CELEBS * G_PER_ID} celebrity + {N_DISTRACTORS} SFHQ)")


def embed(paths):
    feats = []
    for i in range(0, len(paths), BATCH):
        batch = []
        for p in paths[i:i + BATCH]:
            try:
                batch.append(preprocess(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"  skipping corrupted {p}: {e}")
                batch.append(torch.zeros(3, 224, 224))
        with torch.no_grad():
            f = model.encode_image(torch.stack(batch).to(device))
        feats.append(f.float().cpu())
        print(f"  embedded {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    feats = torch.cat(feats)
    return torch.nn.functional.normalize(feats, p=2, dim=1)


print("Embedding gallery...")
g_feats = embed([p for _, p in gallery])
print("Embedding queries...")
q_feats = embed([p for _, p in queries])

sim = q_feats @ g_feats.T
topk = sim.topk(10, dim=1)

# ---------------------------------------------------------------- metrics
top1 = top5 = top10 = 0
rows = []
for qi, (q_ident, q_path) in enumerate(queries):
    idxs = topk.indices[qi].tolist()
    sims = topk.values[qi].tolist()
    ranked_idents = [gallery[g][0] for g in idxs]
    hit_rank = next(
        (r for r, ident in enumerate(ranked_idents) if ident == q_ident), None
    )
    if hit_rank == 0:
        top1 += 1
    if hit_rank is not None and hit_rank < 5:
        top5 += 1
    if hit_rank is not None:
        top10 += 1
    rows.append({
        "query_identity": q_ident,
        "query_path": q_path,
        "hit_rank": hit_rank,          # 0-based; None = not in top-10
        "top_sims": sims,
        "top_paths": [gallery[g][1] for g in idxs],
        "top_idents": ranked_idents,
    })

n = len(queries)
acc1, acc5, acc10 = top1 / n, top5 / n, top10 / n
score = acc1 * 600 + acc5 * 300 + acc10 * 100
summary = {
    "n_queries": n, "n_gallery": len(gallery),
    "n_celebs": N_CELEBS, "g_per_id": G_PER_ID,
    "n_distractors": N_DISTRACTORS, "seed": SEED,
    "top1": acc1, "top5": acc5, "top10": acc10, "score": score,
}
print(json.dumps(summary, indent=2))
with open(OUT_JSON, "w") as fh:
    json.dump({"summary": summary, "rows": rows}, fh, indent=2)

# ---------------------------------------------------------------- figure
# pick rows: 2 clear successes (hit_rank 0), 1 ambiguous (hit in ranks 1-9),
# 2 failures (no hit in top-10)
succ = [r for r in rows if r["hit_rank"] == 0][:2]
ambi = [r for r in rows if r["hit_rank"] not in (0, None)][:1]
fail = [r for r in rows if r["hit_rank"] is None][:2]
chosen = succ + ambi + fail
print(f"Figure rows: {len(succ)} success, {len(ambi)} ambiguous, {len(fail)} failure")

THUMB, PAD, LABEL_H = 160, 8, 22
K = 5
W = PAD + (THUMB + PAD) * (K + 1) + 40
H = (THUMB + LABEL_H + PAD) * len(chosen) + PAD
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()


def thumb(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((THUMB, THUMB))
    bg = Image.new("RGB", (THUMB, THUMB), "white")
    bg.paste(im, ((THUMB - im.width) // 2, (THUMB - im.height) // 2))
    return bg


for ri, row in enumerate(chosen):
    y = PAD + ri * (THUMB + LABEL_H + PAD)
    canvas.paste(thumb(row["query_path"]), (PAD, y))
    draw.text((PAD, y + THUMB + 3), "QUERY", fill="black", font=font)
    for ci in range(K):
        x = PAD + (THUMB + PAD) * (ci + 1) + 40
        canvas.paste(thumb(row["top_paths"][ci]), (x, y))
        correct = row["top_idents"][ci] == row["query_identity"]
        col = (0, 140, 0) if correct else (190, 0, 0)
        mark = "MATCH" if correct else (
            "synthetic" if row["top_idents"][ci].startswith("sfhq") else "wrong"
        )
        draw.rectangle([x - 2, y - 2, x + THUMB + 1, y + THUMB + 1],
                       outline=col, width=3)
        draw.text((x, y + THUMB + 3),
                  f"#{ci + 1}  sim={row['top_sims'][ci]:.2f}  {mark}",
                  fill=col, font=font)

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
canvas.save(OUT_FIG)
print(f"Saved figure: {OUT_FIG}")
