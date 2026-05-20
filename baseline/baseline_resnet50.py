import os           # Libreria per interagire con il sistema operativo (percorsi, file, cartelle)
import json         # Libreria per serializzare/deserializzare dati in formato JSON
from PIL import Image        # PIL (Pillow): libreria per aprire e manipolare immagini
import requests              # Libreria per fare richieste HTTP (es. POST al server)
import torch                 # PyTorch: framework per il deep learning e i tensori
from torchvision.models import resnet50, ResNet50_Weights  # Importa il modello ResNet50 e i suoi pesi pre-addestrati

# --- Selezione del dispositivo di calcolo ---
if torch.cuda.is_available():           # Controlla se è disponibile una GPU NVIDIA (CUDA)
    device = torch.device("cuda")       # Se sì, usa la GPU NVIDIA
elif torch.backends.mps.is_available(): # Altrimenti controlla se è disponibile una GPU Apple Silicon (MPS)
    device = torch.device("mps")        # Se sì, usa la GPU Apple
else:
    device = torch.device("cpu")        # Altrimenti usa la CPU (più lenta)
print(f"Using device: {device}")        # Stampa il dispositivo scelto


# --- Funzione per inviare i risultati al server ---
def submit(results, groupname, url):
    res = {}                            # Crea un dizionario vuoto per la risposta
    res['groupname'] = groupname        # Aggiunge il nome del gruppo al dizionario
    res['images'] = results             # Aggiunge i risultati (query → top-k gallery) al dizionario
    res = json.dumps(res)               # Converte il dizionario in una stringa JSON
    response = requests.post(url, res)  # Invia la stringa JSON via HTTP POST all'URL del server
    try:
        result = json.loads(response.text)          # Prova a decodificare la risposta JSON del server
        print(f"accuracy is {result['accuracy']}")  # Stampa l'accuratezza restituita dal server
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")            # Se la risposta non è JSON valido, stampa l'errore


# --- Funzione per estrarre le feature in batch ---
def batching(images, batch_size=32):
    features = []                                        # Lista che conterrà i tensori di feature
    for i in range(0, len(images), batch_size):          # Itera sulle immagini a gruppi di batch_size
        tmp_images = images[i:i+batch_size]              # Prende il sotto-insieme di immagini del batch corrente
        inputs = torch.stack(
            [preprocess(img.convert("RGB")) for img in tmp_images]  # Converte ogni immagine in RGB e applica il preprocessing (resize, normalizzazione, ecc.)
        ).to(device)                                     # Impila i tensori in un unico batch e lo sposta sul dispositivo (GPU/CPU)
        with torch.no_grad():                            # Disabilita il calcolo del gradiente (non stiamo addestrando, solo inferendo)
            tmp_features = model(inputs)                 # Passa il batch attraverso il modello → ottiene i vettori di feature
            features.append(tmp_features)                # Aggiunge le feature del batch alla lista
    return torch.cat(features, dim=0)                   # Concatena tutti i batch in un unico tensore [N, feature_dim]


# --- Definizione dei percorsi delle cartelle ---
data_folder = "data"     # Cartella radice del dataset
query_folder = os.path.join(data_folder, "query")           # Sottocartella con le immagini di query
gallery_folder = os.path.join(data_folder, "gallery")       # Sottocartella con le immagini della gallery

# --- Liste per memorizzare immagini e nomi dei file ---
query_images = []       # Conterrà gli oggetti PIL delle immagini di query
query_filenames = []    # Conterrà i nomi dei file delle immagini di query
gallery_images = []     # Conterrà gli oggetti PIL delle immagini della gallery
gallery_filenames = []  # Conterrà i nomi dei file delle immagini della gallery

# --- Caricamento delle immagini di query ---
for filename in os.listdir(query_folder):                   # Itera su tutti i file nella cartella query
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Filtra solo i file immagine
        img_path = os.path.join(query_folder, filename)     # Costruisce il percorso completo del file
        query_filenames.append(filename)                     # Salva il nome del file
        img = Image.open(img_path)                          # Apre l'immagine con PIL
        query_images.append(img)                            # Aggiunge l'immagine alla lista

# --- Caricamento delle immagini della gallery (stesso procedimento) ---
for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(gallery_folder, filename)
        gallery_filenames.append(filename)
        img = Image.open(img_path)
        gallery_images.append(img)

# Stampa quante immagini sono state caricate in ciascuna cartella
print(f"Number of images in query folder: {len(query_images)}")
print(f"Number of images in gallery folder: {len(gallery_images)}")


# --- Configurazione del modello ResNet50 ---
weights = ResNet50_Weights.IMAGENET1K_V2          # Seleziona i pesi pre-addestrati su ImageNet (versione V2, più accurata)
model = resnet50(weights=weights)                  # Crea il modello ResNet50 caricando i pesi pre-addestrati
model.fc = torch.nn.Identity()                    # Sostituisce il layer finale (classificatore) con Identity:
                                                  # così l'output è il vettore di feature a 2048 dimensioni, non le 1000 classi ImageNet
model = model.to(device)                          # Sposta il modello sul dispositivo scelto (GPU/CPU)
preprocess = weights.transforms()                 # Ottiene la pipeline di preprocessing associata ai pesi (resize a 224x224, normalizzazione ImageNet, ecc.)
model.eval()                                      # Mette il modello in modalità inferenza (disabilita dropout, batch norm in training mode, ecc.)

# --- Estrazione delle feature ---
print("Processing query images...")
query_features = batching(query_images, batch_size=16)      # Estrae i vettori di feature per tutte le query (batch piccolo, probabilmente poche immagini)
print("Processing gallery images...")
gallery_features = batching(gallery_images, batch_size=256) # Estrae i vettori di feature per tutta la gallery (batch grande per efficienza)

# --- Normalizzazione L2 dei vettori di feature ---
print("Normalizing features...")
query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)     # Normalizza ogni vettore query a lunghezza 1 (norma L2)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1) # Normalizza ogni vettore gallery a lunghezza 1
# Dopo la normalizzazione, il prodotto scalare equivale alla cosine similarity

# --- Calcolo della matrice di similarità coseno ---
print("Computing cosine similarity matrix...")
similarity_matrix = torch.matmul(query_features, gallery_features.T)
# Moltiplica query [N_q, 2048] × gallery_trasposta [2048, N_g] → matrice [N_q, N_g]
# Ogni cella (i, j) contiene la similarità coseno tra la query i e l'immagine gallery j

# --- Recupero dei top-10 match per ogni query ---
print("Getting top 10 matches for each query...")
top_k = min(10, len(gallery_filenames))                                                             # Numero di risultati da restituire per ogni query
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)       # Per ogni query, trova gli indici delle top_k gallery più simili
                                                                        # Il "_" scarta i valori di similarità, teniamo solo gli indici

# --- Conversione degli indici nei nomi dei file corrispondenti ---
top_k_filenames = []                                                    # Lista di liste: per ogni query, i nomi dei top-k file gallery
for i in range(top_k_indices.shape[0]):                                 # Itera su ogni query
    top_k_filenames.append(
        [gallery_filenames[idx] for idx in top_k_indices[i]]           # Mappa ogni indice al nome del file gallery corrispondente
    )

# --- Stampa i risultati a schermo ---
for i, query_filename in enumerate(query_filenames):
    print(f"Top {top_k} matches for {query_filename}:")                 # Intestazione per la query corrente
    for j, gallery_filename in enumerate(top_k_filenames[i]):
        print(f"  {j+1}: {gallery_filename}")                          # Stampa ogni match con la sua posizione nel ranking

# --- Salva i risultati in un dizionario ---
results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = top_k_filenames[i]    # Associa a ogni nome di file query la lista dei suoi top-k match nella gallery

# --- Invia i risultati al server di valutazione ---
submit(results=results, groupname="trade-off", url="http://localhost:3001/retrieval/")
# Chiama la funzione submit con: i risultati, il nome del gruppo, e l'URL del server locale