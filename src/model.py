import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ADClassifier2D(nn.Module):
    def __init__(self, num_classes=3):
        super(ADClassifier2D, self).__init__()

        # Use torchvision's ResNet18 as backbone, adapted for single channel
        self.backbone = resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace final layer for our classification
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for Grad-CAM
        """
        # Forward pass through layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feature_maps = self.backbone.layer4(x)  # Last conv layer

        x = self.backbone.avgpool(feature_maps)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x, feature_maps

def create_model(num_classes=3):
    """
    Create the 2D CNN model for AD classification.
    """
    model = ADClassifier2D(num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(model)

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 224, 224)  # Typical image size
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")