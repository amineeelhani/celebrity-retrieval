import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import clip
from PIL import Image
from utils.submit import submit

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model_b, preprocess_b = clip.load("ViT-B/32", device)
model_l, preprocess_l = clip.load("ViT-L/14", device)
model_b.eval()
model_l.eval()

data_folder = "data"
query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

query_images = []
query_filenames = []
gallery_images = []
gallery_filenames = []

for filename in os.listdir(query_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(query_folder, filename)
        query_filenames.append(filename)
        query_images.append(Image.open(img_path).copy())

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(gallery_folder, filename)
        gallery_filenames.append(filename)
        gallery_images.append(Image.open(img_path).copy())

print(f"Query images: {len(query_images)}")
print(f"Gallery images: {len(gallery_images)}")

def extract_features(images, batch_size=32):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs_b = torch.stack([preprocess_b(img.convert("RGB")) for img in batch]).to(device)
        inputs_l = torch.stack([preprocess_l(img.convert("RGB")) for img in batch]).to(device)
        with torch.no_grad():
            feat_b = model_b.encode_image(inputs_b).float()
            feat_l = model_l.encode_image(inputs_l).float()
        feat_b = torch.nn.functional.normalize(feat_b, p=2, dim=1)
        feat_l = torch.nn.functional.normalize(feat_l, p=2, dim=1)
        feat = torch.cat([feat_b, feat_l], dim=1)
        features.append(feat)
    return torch.cat(features, dim=0)

print("Extracting query features...")
query_features = extract_features(query_images)
print("Extracting gallery features...")
gallery_features = extract_features(gallery_images)

query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)

similarity_matrix = torch.matmul(query_features, gallery_features.T)

top_k = min(10, len(gallery_images))
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = [
        gallery_filenames[idx] for idx in top_k_indices[i]
    ]

submit(results=results, groupname="trade-off", url="http://videosim.disi.unitn.it:3001/retrieval/")
