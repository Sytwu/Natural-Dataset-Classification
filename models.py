import torch
import torch.nn as nn
import torchvision.models as models
from resnest.torch import resnest200


class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        path = "pytorch/vision:v0.10.0"
        self.resnet50 = torch.hub.load(path, "resnet50", weights=True)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def get_parameter_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x):
        return self.resnet50(x)


class ResNeXt101(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNeXt101, self).__init__()
        self.resnext = models.resnext101_64x4d(pretrained=True)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)

    def get_parameter_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x):
        return self.resnext(x)


class ResNeSt200(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNeSt200, self).__init__()
        self.resnest = resnest200(pretrained=True)
        in_features = self.resnest.fc.in_features
        self.resnest.fc = nn.Linear(in_features, num_classes)

    def get_parameter_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def forward(self, x):
        return self.resnest(x)


def get_model(model_name, num_classes):
    if model_name == "ResNeSt200":
        return ResNeSt200(num_classes=num_classes)
    elif model_name == "ResNeXt101":
        return ResNeXt101(num_classes=num_classes)
    elif model_name == "ResNet50":
        return ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    # Example usage: Initialize and print parameter size.
    model = get_model("ResNet50", num_classes=100)
    model.get_parameter_size()
