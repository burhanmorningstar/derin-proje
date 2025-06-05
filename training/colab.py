""" 1. Başlangıç: İlk Model Eğitimi (Google Colab)
Veri Seti: RWF-2000 ve Real Life Violence Situations gibi şiddet tespitine yönelik video veri setleri kullanıldı.

Ön İşleme:

Videolardan kareler (frames) çıkarıldı.

Her videoyu frame’lere bölüp klasörlerde Violence ve NonViolence olarak ayırdık.

Model Mimarisi:

CNN (MobileNetV2) + LSTM yapısı.

CNN: Görsel özellikleri çıkarıyor.

LSTM: Zaman içinde bu özellikleri analiz ediyor.

Model Eğitimi:

Google Colab üzerinde PyTorch kullanılarak eğitim yapıldı.

Model ağırlıkları SonDeneme.pth olarak kaydedildi.

🖥 2. Local Ortama Geçiş (VS Code)
Model Ağırlıklarını İndirme:

Colab’da eğitilen SonDeneme.pth dosyası bilgisayara indirildi.

Local klasör yapısına uygun olarak konumlandırıldı.

Yeni Dataset Hazırlığı:

frames_split/train ve frames_split/val klasörleri oluşturuldu.

Videolar karelere ayrılarak eğitim/validasyon için frame dizileri oluşturuldu.

Her frame dizisi bir video sekansını temsil etti.

🔧 3. Fine-Tuning (Modeli Yerel Olarak Güncelleme)
Kod Güncellemeleri:

train_finetune2.py adında bir eğitim dosyası oluşturuldu.

SonDeneme.pth ağırlıkları yüklendi.

Eğitim verileriyle birkaç epoch boyunca fine-tuning yapıldı.

Yeni ağırlıklar FineTuned_SonDeneme.pth olarak kaydedildi.

Özellikler:

Dropout ve Early Stopping eklendi.

Eğitim süreci sırasında loss ve accuracy görselleştirildi.

Modelin overfit olmasını engellemek için patience ile durduruldu.

🎥 4. Test & Görsel Arayüz (PyQt5 ile Canlı Video Analizi)
test.py dosyası oluşturuldu.

PyQt5 arayüzü geliştirildi:

Videoyu canlı oynatıyor.

Her 10 frame'lik dizi için modelden tahmin alıyor.

Tahmin yüzdelerini anlık olarak kullanıcıya gösteriyor.

Tahmin Hesaplama:

Model her 10 karede bir tahmin yapıyor.

Tüm tahminler toplanıyor (self.total_preds)

Şiddet oranı hesaplanıyor → Eğer %50'den fazlaysa "Violence" diyor.

🔍 5. Sonuçlar ve Gözlemler
Yüksek doğrulukta tahminler elde edildi.

Anlık tahminlerde esneklik ve görsel çıktı sağlandı.

Model çok hassas olduğu için saniyelik analiz bile yapılabildi.

Şu anda model herhangi bir video dosyasını okuyup şiddet oranını yüzdesel olarak verebiliyor. """

"""
kullandığımız datasetler burada :




!chmod 600 ~/.kaggle/kaggle.json

# 📁 Datasetleri indir
!kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset
!kaggle datasets download -d vulamnguyen/rwf2000

# 🎯 Dosyaları çıkar
!unzip -q real-life-violence-situations-dataset.zip -d data/
!unzip -q rwf2000.zip -d data/"""


import os
import cv2
from tqdm import tqdm

def extract_frames_from_videos(input_dir, output_dir, label, frame_drop=3, resize_to=(320, 240)):
    os.makedirs(output_dir, exist_ok=True)
    videos = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for video in tqdm(videos, desc=f"Processing {label} videos"):
        video_path = os.path.join(input_dir, video)
        cap = cv2.VideoCapture(video_path)

        count = 0
        frame_index = 0
        vidname = os.path.splitext(video)[0]
        out_folder = os.path.join(output_dir, f"{label}_{vidname}")
        os.makedirs(out_folder, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_drop == 0:
                # Frame'i küçült
                frame = cv2.resize(frame, resize_to)

                # İleri aşamada detection yapılabilsin diye kutu yerleştirme alanı ayır
                frame_path = os.path.join(out_folder, f"{count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                count += 1

            frame_index += 1

        cap.release()

# 🔄 Frame extraction (optimize edilmiş)
extract_frames_from_videos("data/Real Life Violence Dataset/Violence", "frames/violence", "violence")
extract_frames_from_videos("data/Real Life Violence Dataset/NonViolence", "frames/nonviolence", "nonviolence")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import random
import os

# 📊 Dataset class (frame listeleri + etiket)
class ViolenceDataset(Dataset):
    def _init_(self, root_dir, label, transform=None, max_frames=30):
        self.samples = []
        self.transform = transform
        self.max_frames = max_frames
        self.label = label
        folders = glob.glob(os.path.join(root_dir, "*"))
        for folder in folders:
            frame_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            if len(frame_paths) >= max_frames:
                self.samples.append((frame_paths[:max_frames], label))

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for path in frame_paths:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        return torch.stack(frames), label  # [T, C, H, W], label

# 🔍 Görsel dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 📂 Datasetleri yükle
violence_ds = ViolenceDataset("frames/violence", 1, transform)
nonviolence_ds = ViolenceDataset("frames/nonviolence", 0, transform)

# 🔀 Datasetleri birleştir ve karıştır
full_samples = violence_ds.samples + nonviolence_ds.samples
random.shuffle(full_samples)

# 🔁 Custom CombinedDataset class
class CombinedDataset(Dataset):
    def _init_(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        frame_paths, label = self.samples[idx]
        frames = []
        for path in frame_paths:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        return torch.stack(frames), label

# 🔀 Train-validation ayır
train_size = int(0.8 * len(full_samples))
train_ds = CombinedDataset(full_samples[:train_size], transform)
val_ds = CombinedDataset(full_samples[train_size:], transform)

# 📦 DataLoader
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 🧠 CNN + LSTM Model Sınıfı
class CNNLSTM(nn.Module):
    def _init_(self):
        super()._init_()
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.features))  # [B, 1280, 7, 7]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))               # [B, 1280, 1, 1]
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).view(B, T, -1)  # [B, T, 1280]
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# ✅ Eğitim Ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5
BATCH_SIZE = 2

# ❗ Not: train_loader, val_loader ve test_loader daha önce tanımlanmış olmalı

# 📈 Eğitim Takibi
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    true_labels, pred_labels = [], []

    loop = tqdm(train_loader, desc=f"🚂 Epoch {epoch+1}/{EPOCHS}", leave=False)
    for videos, labels in loop:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds)

        loop.set_postfix(loss=loss.item())

    train_acc = accuracy_score(true_labels, pred_labels)
    train_losses.append(total_loss)
    train_accuracies.append(train_acc)

    # 🔍 Validation
    model.eval()
    val_loss, val_true, val_pred = 0, [], []
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(preds)

    val_acc = accuracy_score(val_true, val_pred)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"📦 Epoch {epoch+1} - Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# 📊 Eğitim Sonrası Grafik
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.show()

import os
import cv2

def extract_frames_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    for video in video_files:
        cap = cv2.VideoCapture(os.path.join(input_folder, video))
        count = 0
        name = os.path.splitext(video)[0]
        save_path = os.path.join(output_folder, name)
        os.makedirs(save_path, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(save_path, f"{name}_{count:04d}.jpg"), frame)
            count += 1

        cap.release()
        print(f"📁 {video} için {count} frame çıkarıldı → {save_path}")


# 📁 Klasör yollarını ayarla
violence_path = "Data Topluyorum/Violence"
nonviolence_path = "Data Topluyorum/NonViolence"

output_violence = "frames/Violence"
output_nonviolence = "frames/NonViolence"

# 🔁 Frame'leri çıkar
extract_frames_from_folder(violence_path, output_violence)
extract_frames_from_folder(nonviolence_path, output_nonviolence)

print("✅ Tüm videolardan frame çıkarma işlemi tamamlandı.")


import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import random

# 🔧 Ayarlar
sequence_length = 10
batch_size = 2
epochs = 20
learning_rate = 1e-4
patience = 3  # early stopping için sabır

# 📦 CNN + LSTM Modeli (Dropout eklendi, num_layers=1 olarak düzeltildi)
class CNNLSTM(nn.Module):
    def _init_(self):
        super()._init_()
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.features))
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).view(B, T, -1)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 🧠 Dataset
class VideoDataset(Dataset):
    def _init_(self, root_dir, sequence_length=10):
        self.samples = []
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        for label, class_dir in enumerate(['NonViolence', 'Violence']):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.exists(class_path):
                print(f"[UYARI] Klasör bulunamadı: {class_path}")
                continue
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if os.path.isdir(video_path):
                    frames = sorted(os.listdir(video_path))
                    if len(frames) >= sequence_length:
                        self.samples.append((video_path, frames, label))

        print(f"[INFO] {root_dir} klasöründen toplam {len(self.samples)} video örneği bulundu.")

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        video_path, frames, label = self.samples[idx]
        total_frames = len(frames)
        indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        selected_frames = [frames[i] for i in indices]
        images = []
        for frame_name in selected_frames:
            frame_path = os.path.join(video_path, frame_name)
            image = Image.open(frame_path).convert("RGB")
            images.append(self.transform(image))
        images = torch.stack(images)  # [T, C, H, W]
        return images, label

# ✅ Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Kullanılan cihaz:", device)

# ✅ Veriyi yükle
train_dataset = VideoDataset("frames_split/train", sequence_length)
val_dataset = VideoDataset("frames_split/val", sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ✅ Model yükle (önceden eğitilmiş, num_layers=1 ile uyumlu)
model = CNNLSTM().to(device)
model.load_state_dict(torch.load("SonDeneme.pth", map_location=device))
print("✅ SonDeneme.pth başarıyla yüklendi!")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Eğitim döngüsü (Early stopping dahil)
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    true_labels, pred_labels = [], []

    loop = tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{epochs}", leave=False)
    for videos, labels in loop:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds)
        loop.set_postfix(loss=loss.item())

    train_acc = accuracy_score(true_labels, pred_labels)
    train_losses.append(running_loss)
    train_accuracies.append(train_acc)

    # 🔍 Validation
    model.eval()
    val_loss = 0.0
    val_true, val_pred = [], []
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_true.extend(labels.cpu().numpy())
            val_pred.extend(preds)

    val_acc = accuracy_score(val_true, val_pred)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"📦 Epoch {epoch+1} - Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ✅ Early stopping kontrolü
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "FineTuned_SonDeneme2.pth")
        print("✅ Yeni en iyi model kaydedildi!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping devreye girdi!")
            break

# ✅ Grafik
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.show()