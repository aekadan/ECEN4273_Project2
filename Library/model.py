
import torch
import torch.nn as nn
from torchvision import models

#Create a ResNet-18 model for multi-label image classification.
def create_model(num_classes=8, pretrained=True):

    # Load ResNet-18 with pretrained ImageNet weights
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the original classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

#Return independent probabilities for each image class.
def predict(model, images, device="cpu"):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits)

    return probabilities
