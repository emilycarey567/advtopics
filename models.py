import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. PILOTNET (Baseline)
# ------------------------------------------------------------
# Implements the original NVIDIA PilotNet architecture:
# 5 conv layers (strided, no pooling) + 5 fully connected layers.
# Input: 66x200x3 (YUV or RGB)
# Reference: Bojarski et al. (2016) - End to End Learning for Self-Driving Cars
# ============================================================
class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        # Fully connected head
        self.fc1 = nn.Linear(64 * 1 * 18, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = self.drop(x)
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        return self.out(x)


# ============================================================
# 2. PILOTNET + SWISH ACTIVATION
# ------------------------------------------------------------
# Same as PilotNet but uses Swish activation (x * sigmoid(x))
# for smoother gradients and potentially better generalization.
# ============================================================
class PilotNetSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(64 * 1 * 18, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

        self.drop = nn.Dropout(p=0.1)
        self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        act = self.swish
        x = act(self.conv1(x))
        x = act(self.conv2(x))
        x = act(self.conv3(x))
        x = act(self.conv4(x))
        x = act(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = act(self.fc1(x))
        x = self.drop(x)
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        x = act(self.fc4(x))
        return self.out(x)


# ============================================================
# 3. RESNET-PILOT
# ------------------------------------------------------------
# Compact ResNet-inspired variant for deeper feature extraction.
# Includes residual skip connections.
# ============================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetPilot(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = BasicBlock(32, 64, stride=2)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 128, stride=1)

        self.fc1 = nn.Linear(128 * 4 * 12, 256)
        self.fc2 = nn.Linear(256, 50)
        self.fc3 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 1)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


# ============================================================
# 4. VGG-PILOT (Dynamic Fix)
# ------------------------------------------------------------
# VGG-style architecture for behavioural cloning.
# Uses 3x3 conv blocks, pooling, and dynamically infers FC input size.
# ============================================================
class VGGPilot(nn.Module):
    """
    Compact VGG-style network for behavioural cloning.
    Automatically infers the FC input size from the actual conv output.
    """
    def __init__(self):
        super().__init__()

        # Confirm this is the latest version
        print("[VGGPilot] ✅ Loaded dynamic version (will infer flatten size automatically)")

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier built lazily on first forward()
        self.classifier = None

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        if self.classifier is None:
            in_features = x.shape[1]
            print(f"[VGGPilot] 🔧 Building classifier dynamically with in_features={in_features}")
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 10),
                nn.ReLU(inplace=True),
                nn.Linear(10, 1)
            ).to(x.device)
        return self.classifier(x)
