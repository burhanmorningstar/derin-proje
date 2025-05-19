# training/data_prep.py

import os, urllib.request, zipfile, csv
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

DATA_ZIP    = "hasy-data.zip"
LABEL_CSV   = "hasy-data-labels.csv"
IMG_URL     = "https://zenodo.org/record/259444/files/hasy-data.zip?download=1"
LABEL_URL   = "https://zenodo.org/record/259444/files/hasy-data-labels.csv?download=1"
EXTRACT_DIR = "hasy-data"

def download_and_extract():
    if not os.path.exists(DATA_ZIP):
        print("📥 Downloading images…")
        urllib.request.urlretrieve(IMG_URL, DATA_ZIP)
    with zipfile.ZipFile(DATA_ZIP, 'r') as z:
        z.extractall(".")
    print(f"✅ Extracted images to ./{EXTRACT_DIR}/")

    if not os.path.exists(LABEL_CSV):
        print("📥 Downloading labels…")
        urllib.request.urlretrieve(LABEL_URL, LABEL_CSV)
    print(f"✅ Labels saved as ./{LABEL_CSV}")

class HASYDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform or transforms.Compose([
    transforms.RandomRotation(15),                       # Rastgele ±15 derece döndürme
    transforms.RandomAffine(degrees=0, translate=(0.15,0.15)),  # Rastgele öteleme (%15)
    transforms.Grayscale(num_output_channels=1),         # Gri tona çevir (tek kanal)
    transforms.Resize((28,28)),                          # 28x28 boyutuna sabitle
    transforms.ToTensor(),                               # Tensor formatına çevir
    transforms.Normalize((0.5,), (0.5,))                 # Normalizasyon
])


    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        return self.transform(img), self.labels[idx]

def prepare_data(test_size=0.2):
    download_and_extract()

    # 1) CSV’den raw path + raw label oku
    raw_paths, raw_labels = [], []
    with open(LABEL_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # path örn: "hasy-data/983.png"
            fn = os.path.join(EXTRACT_DIR, os.path.basename(row['path']))
            raw_paths.append(fn)
            raw_labels.append(int(row['symbol_id']))

    # 2) unique symbol_id’leri küçükten büyüğe sırala ve 0…N-1 index ata
    unique_ids = sorted(set(raw_labels))
    id2idx = {orig: i for i, orig in enumerate(unique_ids)}
    mapped_labels = [id2idx[lab] for lab in raw_labels]

    # 3) train/val split (stratify mapped_labels)
    indices = list(range(len(raw_paths)))
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size,
        stratify=mapped_labels, random_state=42
    )

    # 4) böl ve döndür
    train_files  = [raw_paths[i]  for i in train_idx]
    train_labels = [mapped_labels[i] for i in train_idx]
    val_files    = [raw_paths[i]  for i in val_idx]
    val_labels   = [mapped_labels[i] for i in val_idx]

    return (train_files, train_labels), (val_files, val_labels)

if __name__ == "__main__":
    (tr_f, tr_l), (vl_f, vl_l) = prepare_data()
    print(f"🗂 Train: {len(tr_f)} samples, Val: {len(vl_f)} samples")
    print(f"🔢 Num classes = {len(set(tr_l))}")
