import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch
from utils.face_detection import load_mtcnn
from torchvision import transforms

# percorso input e output
input_folder = "/path/to/vggface2"        # cartella originale
output_folder = "/path/to/vggface2_faces" # cartella output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = load_mtcnn(device)

# fallback transform se MTCNN non trova il viso
fallback = transforms.Resize((224, 224))

skipped = 0
processed = 0

for identity in os.listdir(input_folder):
    identity_in  = os.path.join(input_folder, identity)
    identity_out = os.path.join(output_folder, identity)
    os.makedirs(identity_out, exist_ok=True)

    for filename in os.listdir(identity_in):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(identity_in, filename)
        out_path = os.path.join(identity_out, filename)

        try:
            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)

            if face is not None:
                # MTCNN restituisce tensore → convertiamo in PIL per salvare
                face_pil = transforms.ToPILImage()(face.clamp(0, 1))
                face_pil.save(out_path)
                processed += 1
            else:
                # fallback — salva immagine ridimensionata
                fallback_img = fallback(img)
                fallback_img.save(out_path)
                skipped += 1

        except Exception as e:
            print(f"Errore su {filename}: {e}")
            skipped += 1

print(f"Processed: {processed}")
print(f"Fallback:  {skipped}")