import torch
import torch.nn as nn

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetB4, self).__init__()
        
        # Simple initial convolution
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Block 1: Example with Depthwise Separable Convolution
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.depthwise = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96)
        self.pointwise = nn.Conv2d(96, 192, kernel_size=1)

        # Adaptive pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(192, 1024)  # Adjusted after convolution output size
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Initial Conv and Max Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 1: Convolution, Depthwise Separable Conv, and Pointwise Conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Depthwise Separable Convolution
        x = self.depthwise(x)
        x = self.pointwise(x)

        # Global average pooling to reduce dimensions
        x = self.global_pool(x)

        # Flatten the tensor and pass through the fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
