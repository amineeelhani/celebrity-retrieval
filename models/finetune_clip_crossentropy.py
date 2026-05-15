import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import clip
from PIL import Image
from torch import nn                          # CrossEntropyLoss
from torch import optim                       # ottimizzatore
from torch.utils.data import DataLoader       # dataloader
from torchvision import datasets, transforms  # dataset e augmentation
from utils.face_detection import load_mtcnn, detect_face


TRAINING_MODE = "vggface2"
if TRAINING_MODE == "vggface2":
    data_folder = "/path/to/vggface2"
else:
    data_folder = "data/train"   

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")



model, preprocess = clip.load("ViT-B/32",device)
model.eval()
mtcnn = load_mtcnn(device)

#freeze CLIP weights
for param in model.parameters():
    param.requires_grad = False
#unfreeze last 4 layers if vggface2
if TRAINING_MODE == "vggface2":
    for layer in model.visual.transformer.resblocks[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

N_IDENTITY= len(os.listdir(data_folder))
classification_head = nn.Linear(512, N_IDENTITY).to(device)
#Il classification head serve ogni volta che vuoi addestrare un modello 
# con una loss di classificazione su un numero fisso di identità. 
# Senza di esso CLIP non sa quante identità 
# esistono nel tuo dataset e non può calcolare la loss.