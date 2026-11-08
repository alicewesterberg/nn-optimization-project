import torch
import torch.nn as nn

def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            
            
        )
        
    def forward(self, x):
        x = self.features(x)       # (B, 3, 32, 32) -> (B, 128, 4, 4)
        x = self.classifier(x)     # (B, 128, 4, 4) -> (B, 10)
        return x

def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"Okänd modell: {name}")


