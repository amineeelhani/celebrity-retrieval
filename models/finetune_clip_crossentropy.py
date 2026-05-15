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
    data_folder = "/path/to/vggface2_faces"
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

#DATA AUGMENTATION
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 80% training, 20% validation
train_dataset_full = datasets.ImageFolder(data_folder, transform=train_transform)
val_dataset_full   = datasets.ImageFolder(data_folder, transform=val_transform)

n_val   = int(0.2 * len(train_dataset_full))
n_train = len(train_dataset_full) - n_val

# stessi indici per entrambi
indices = torch.randperm(len(train_dataset_full)).tolist()
train_dataset = torch.utils.data.Subset(train_dataset_full, indices[:n_train])
val_dataset   = torch.utils.data.Subset(val_dataset_full,   indices[n_train:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

print(f"Training images:   {n_train}")
print(f"Validation images: {n_val}")

#OPTIMIZATION
if TRAINING_MODE == "vggface2":
    optimizer = optim.Adam([
        {"params": classification_head.parameters(), "lr": 1e-3},
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": 1e-5}
    ])
else:  # dati del prof
    optimizer = optim.Adam([
        {"params": classification_head.parameters(), "lr": 1e-3}
    ])

criterion = nn.CrossEntropyLoss()

#TRAINING LOOP
NUM_EPOCHS = 10
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.eval()
    classification_head.train()
    #Perché alla fine di ogni epoca fate la validation e mettete tutto in .eval(). 
    # All'inizio della prossima epoca dovete riportare il classification head in .train():
    train_loss = 0.0 #accumula la loss di ogni batch alla fine dell'epoca dividi per il numero di batch e ottini la loss media del epoca
    train_correct = 0 #conta quante immagini il modello ha classificato correttamente
    train_total = 0 #conta quante immagini totali ha visto il modello
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        #Perché serve? Il modello è sulla GPU — le immagini arrivano dal dataloader sulla CPU. 
        # Devono essere sullo stesso device altrimenti PyTorch crasha.
        optimizer.zero_grad()
        #Ogni batch deve essere indipendente 
        # — zero_grad() azzera i gradienti prima di calcolare quelli del batch corrente. se no si accumulano
        with torch.no_grad():
            features = model.encode_image(images) #(32, 3, 224, 224) → (32, 512)
        features = features.float()
        #CLIP restituisce float16 (metà precisione) per efficienza 
        # CrossEntropy si aspetta float32 (precisione piena) 
        # .float() converte da float16 a float32
        logits = classification_head(features)
        loss = criterion(logits, labels)

        loss.backward() #calcola gradienti
        optimizer.step() # aggiorna pesi

        train_loss += loss.item() #.item() estrae il valore dal tensore
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0) #numero di elementi nella dimensione 0

    train_acc = train_correct/train_total

    #VALIDATION
    model.eval()
    classification_head.eval()
    val_correct = 0
    val_total = 0
        
