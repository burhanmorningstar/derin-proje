# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from data_prep import HASYDataset, prepare_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_model.pth"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=369):  # HASY’de ~369 sembol var
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),    nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),   nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def main(epochs=30, batch_size=256, lr=1e-3):
    (tr_f, tr_l), (vl_f, vl_l) = prepare_data()
    train_ds = HASYDataset(tr_f, tr_l)
    val_ds   = HASYDataset(vl_f, vl_l)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=len(set(tr_l))).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"📝 New best model saved ({best_acc:.4f})")

if __name__ == "__main__":
    main()
