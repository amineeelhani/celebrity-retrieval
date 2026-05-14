import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
from utils.submit import submit, evaluate_local
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)

model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
        img = Image.open(img_path)
        query_images.append(img)

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(gallery_folder,filename)
        gallery_filenames.append(filename)
        img = Image.open(img_path)
        gallery_images.append(img)

print(f"Query images: {len(query_images)}")
print(f"Gallery images: {len(gallery_images)}")

def extract_features(images, batch_size = 32):
    features = []
    for i in range(0,len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = torch.stack([preprocess(img.convert("RGB")) for img in batch]).to(device)
        with torch.no_grad():
            feature = model(inputs)
        features.append(feature)
    return torch.cat(features, dim = 0)

print("Extracting query features...")
query_features = extract_features(query_images)
print("Extracting gallery features...")
gallery_features = extract_features(gallery_images)


query_features = torch.nn.functional.normalize(query_features, p = 2, dim = 1)
gallery_features = torch.nn.functional.normalize(gallery_features, p = 2, dim = 1)

similarity_matrix = torch.matmul(query_features, gallery_features.T)

top_k = min(10, len(gallery_images))
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = [
        gallery_filenames[idx] for idx in top_k_indices[i]
    ]

#submit(results=results, groupname="trade-off", url="http://localhost:3001/retrieval/")

ground_truth = {
    "brad1.jpg": ["brad2.jpg"]
}

evaluate_local(results, ground_truth)