# backend/app/predict.py

import io
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
import numpy as np
from torchvision.transforms.functional import resize, to_tensor, normalize

# same normalization as training
transform = transforms.Compose([
    transforms.RandomRotation(15),                       # Rastgele ±15 derece döndürme
    transforms.RandomAffine(degrees=0, translate=(0.15,0.15)),  # Rastgele öteleme (%15)
    transforms.Grayscale(num_output_channels=1),         # Gri tona çevir (tek kanal)
    transforms.Resize((28,28)),                          # 28x28 boyutuna sabitle
    transforms.ToTensor(),                               # Tensor formatına çevir
    transforms.Normalize((0.5,), (0.5,))                 # Normalizasyon
])



def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")      # PIL image
    arr = np.array(img)                                         # H×W
    # 1) Sembolün bbox'unu bul (siyah pikseller < 250 eşiği)
    ys, xs = np.where(arr < 250)
    if len(xs) == 0 or len(ys) == 0:          # boş görsel
        raise UnidentifiedImageError
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    # 2) Kırp ve kare yastık ekle
    cropped = img.crop((xmin, ymin, xmax+1, ymax+1))            # PIL crop
    # 3) Kare tuval oluştur (padding) – uzun kenarı 22px'e sığacak
    cropped = resize(cropped, 22)                               # küçük kenar ≈22 px
    # 22×22'yi ortalayıp 28×28'e padle
    canvas = Image.new("L", (28,28), 255)                       # beyaz tuval
    offset = ((28-cropped.width)//2, (28-cropped.height)//2)
    canvas.paste(cropped, offset)
    # 4) Tensor + normalize
    tensor = to_tensor(canvas)                                  # [1,28,28]
    tensor = normalize(tensor, (0.5,), (0.5,))
    return tensor

@torch.no_grad()
def model_inference(model, device, image_bytes: bytes, idx2symbol: dict):
    """
    Returns single best (symbol, confidence).
    """
    tensor = preprocess_image(image_bytes).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return idx2symbol[int(idx)], float(conf)
