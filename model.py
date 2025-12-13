"""
Coffee Bean Classification Models
Supports both Custom 3-Layer CNN and ResNet18 Transfer Learning
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class SimpleCNN(nn.Module):
    """
    Custom 3-Layer CNN for Coffee Bean Classification
    Accuracy: 56.41% (60% Arabica, 52.63% Robusta)
    Parameters: 93,954
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Layer 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Layer 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Layer 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Dropout + Classification
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_resnet18_finetuned(num_classes=2, pretrained=True):
    """
    Create ResNet18 model for fine-tuning
    Accuracy: 71.79% (80% Arabica, 63.16% Robusta)
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Load pretrained ImageNet weights (default: True)
    
    Returns:
        model: ResNet18 model with modified final layer
    """
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def freeze_resnet_except_layer4(model):
    """
    Freeze all ResNet layers except layer4 and fc
    Used for fine-tuning with limited data
    
    Args:
        model: ResNet model to freeze
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 (last conv block)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Unfreeze final FC layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def get_model(model_type='resnet18', pretrained=True, num_classes=2):
    """
    Factory function to get the desired model
    
    Args:
        model_type (str): 'simple_cnn' or 'resnet18'
        pretrained (bool): Load pretrained weights (for ResNet18)
        num_classes (int): Number of output classes
    
    Returns:
        model: The requested model
    """
    if model_type == 'simple_cnn':
        return SimpleCNN()
    elif model_type == 'resnet18':
        return create_resnet18_finetuned(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'simple_cnn' or 'resnet18'")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing SimpleCNN...")
    simple_model = SimpleCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    print("\nTesting ResNet18...")
    resnet_model = create_resnet18_finetuned().to(device)
    print(f"Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
    
    print("\nTesting ResNet18 with frozen layers...")
    resnet_model = freeze_resnet_except_layer4(resnet_model)
    trainable = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = resnet_model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
    print("✓ All models working correctly!")
