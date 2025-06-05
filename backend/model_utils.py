# model_utils.py
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.features))
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
