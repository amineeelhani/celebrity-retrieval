import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- ArcFace Loss Implementation ---
class ArcFaceLinear(nn.Module):
    """
    Custom classification head implementing ArcFace loss.
    Replaces nn.Linear with angular margin enforcement.
    """
    def __init__(self, in_features, out_features, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        # weight matrix: one row per identity
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # normalize both features and weights onto unit sphere
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # cosine similarity between each feature and each class center
        cosine = F.linear(features, weight)  # (batch, N_IDENTITY)

        if labels is None:
            # validation mode — no margin, just scale
            return cosine * self.scale

        # training mode — add margin only to correct class angle
        theta = cosine.clamp(-1 + 1e-7, 1 - 1e-7).acos()
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = (theta + self.margin * one_hot).cos() * self.scale
        return logits


# --- Configuration ---
# Change this to switch between training modes
TRAINING_MODE = "competition"   # "competition" = professor's data

if TRAINING_MODE == "competition":
    data_folder = "data/train"
else:
    raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")

# --- Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Load CLIP ViT-L/14 ---
# KEY DIFFERENCE FROM PREVIOUS FILE:
# We use ViT-L/14 (768-dim) instead of ViT-B/32 (512-dim)
# because ViT-L/14 is our best performing retrieval model (666.3/1000)
model, preprocess = clip.load("ViT-L/14", device)
model.eval()

# --- Freeze all CLIP layers first ---
for param in model.parameters():
    param.requires_grad = False

# --- KEY FIX: Unfreeze last 4 transformer layers in ALL modes ---
# In the previous file, competition mode froze everything.
# This meant model.state_dict() was identical to original CLIP.
# Now we unfreeze last 4 layers so they actually get updated.
for layer in model.visual.transformer.resblocks[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable CLIP parameters: {trainable:,}")

# --- Classification Head ---
# KEY DIFFERENCE: 768 input features (ViT-L/14) instead of 512 (ViT-B/32)
N_IDENTITY = len(os.listdir(data_folder))
print(f"Number of identities: {N_IDENTITY}")
classification_head = ArcFaceLinear(768, N_IDENTITY).to(device)

# --- Data Augmentation ---
# Face-specific augmentations to improve robustness to domain shift
# (real photos vs synthetic images)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset and DataLoader ---
train_dataset_full = datasets.ImageFolder(data_folder, transform=train_transform)
val_dataset_full   = datasets.ImageFolder(data_folder, transform=val_transform)

n_val   = int(0.2 * len(train_dataset_full))
n_train = len(train_dataset_full) - n_val

indices = torch.randperm(len(train_dataset_full)).tolist()
train_dataset = torch.utils.data.Subset(train_dataset_full, range(500))#indices[:n_train])
val_dataset   = torch.utils.data.Subset(val_dataset_full,   range(100))#indices[n_train:])

# num_workers=0 to avoid deadlock issues on Azure VM
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

print(f"Training images:   {n_train}")
print(f"Validation images: {n_val}")

# --- Optimizer ---
# Two learning rates:
# - classification head: lr=1e-3 (learning from scratch)
# - CLIP last 4 layers:  lr=1e-5 (fine adjustment, avoid catastrophic forgetting)
optimizer = optim.Adam([
    {"params": classification_head.parameters(), "lr": 1e-3},
    {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": 1e-6}
])

criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
NUM_EPOCHS = 10
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # CLIP in eval mode (stable batch norm), head in train mode
    model.eval()
    classification_head.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Extract features — no grad for frozen layers
        # but grad flows through unfrozen last 4 layers
        #with torch.no_grad():
        features = model.encode_image(images)
        features = features.float()

        # ArcFace needs labels during forward pass to apply angular margin
        logits = classification_head(features, labels)
        loss = criterion(logits, labels)

        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(classification_head.parameters(), max_norm=1.0)

        # verifica gradienti — rimuovete dopo il test
        for layer in model.visual.transformer.resblocks[-4:]:
            for param in layer.parameters():
                if param.grad is not None:
                    print(f"Grad norm: {param.grad.norm().item():.6f}")
                break  # ← dentro il for param
            break  # ← dentro il for layer

        optimizer.step()

        train_loss += loss.item()
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # --- Validation ---
    model.eval()
    classification_head.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = model.encode_image(images).float()
            # No labels in validation — no angular margin
            logits = classification_head(features)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    # Save best model by validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"models/clip_vitl14_arcface_{TRAINING_MODE}.pt")
        print(f" -> Saved best model (val_acc: {val_acc*100:.1f}%)")

print(f"\nTraining complete. Best val_acc: {best_val_acc*100:.1f}%")
print(f"Model saved to: models/clip_vitl14_arcface_{TRAINING_MODE}.pt")