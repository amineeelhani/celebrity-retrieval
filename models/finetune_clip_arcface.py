import torch.nn.functional as F
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

class ArcFaceLinear(nn.Module):
    def __init__(self, in_features, out_features, margin = 0.5, scale = 64):
        super().__init__() ##inizializza la classe padre nn.Module, abilitando tutti i meccanismi di pytorch
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features,in_features)) #dice a PyTorch che questa matrice è un parametro del layer
        #torch.FloatTensor(out_features, in_features) crea una matrice vuota di numeri float 
        # F.linear(features, weight) fa:features × weight.T = (32, 512) × (512, 9000) = (32, 9000)
        nn.init.xavier_uniform_(self.weight)#inizializza i pesi con valori casuali intelligenti non troppo grandi, non troppo piccoli, il training parte in modo stabile
    def forward(self, features, labels = None): #Si chiama automaticamente quando  classification_head(features, labels)
        features = F.normalize(features, p =2, dim=1)
        weight = F.normalize(self.weight, p = 2, dim =1)
        cosine = F.linear(features,weight)
        if labels is None:
            return cosine * self.scale
        else:
            theta = cosine.clamp(-1 + 1e-7, 1- 1e-7).acos() #angolo tra embedding e centro classe
            one_hot = F.one_hot(labels, num_classes= cosine.size(1)).float() #crea una matrice che indica quale è la classe corretta per ogni immagine, serve per aggiungere il margine solo alla classe corretta
            logits = (theta + self.margin*one_hot).cos() * self.scale #.cos riconvertiamo in coseno, perchè CrossEntropy lavora con coseni, non angoli
            return logits


TRAINING_MODE = "vggface2"
if TRAINING_MODE == "vggface2":
    data_folder = "/mnt/vggface2/train"
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
classification_head = ArcFaceLinear(512, N_IDENTITY).to(device)

#DATA AUGMENTATION
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
        logits = classification_head(features,labels) #arcface ha bisogno di sapere quale è la classe corretta durante il calcolo dei logits, non solo dopo
        loss = criterion(logits, labels)

        loss.backward() #calcola gradienti
        optimizer.step() # aggiorna pesi

        train_loss += loss.item() #.item() estrae il valore dal tensore
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0) #numero di elementi nella dimensione 0

    train_acc = train_correct/train_total

    #VALIDATION "il modello che ha imparato sulle immagini di training 
    # funziona anche su immagini che non ha mai visto?"
    model.eval()
    classification_head.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = model.encode_image(images).float()
            logits = classification_head(features)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)
        
        
    val_acc = val_correct/ val_total        
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),f"models/clip_arcface_{TRAINING_MODE}.pt")
        print(f" -> Saved best model (val_acc: {val_acc*100:.1f}%)")