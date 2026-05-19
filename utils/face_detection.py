from facenet_pytorch import MTCNN
import torch

def load_mtcnn(device):
    #return MTCNN(image_size=224, keep_all=False, device=device)
    return None
def detect_face(img, mtcnn, preprocess):
    face = mtcnn(img)
    if face is not None:
        return face
    return preprocess(img.convert("RGB"))