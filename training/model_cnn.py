# training/model_cnn.py
import torch.nn as nn

class PopoCNN(nn.Module):
    """
    Conv(32) → Conv(64) → Conv(96) → FC(128) per PopoDev/ML_CLF_HASYv2
    Input: 1×32×32
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                # 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                # 8×8
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2),                                # 4×4
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                   # 96*4*4 = 1536
            nn.Dropout(0.5),
            nn.Linear(1536, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x): return self.head(self.conv(x))
