import torch.nn as nn
import torchvision


def LinearModel(num_in, num_out):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_in, num_out),
    )


def SingleLayer(num_in, num_out, num_features):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_in, num_features),
        nn.GELU(),
        nn.Linear(num_features, num_out),
    )


def MLP(num_in, num_out, num_features, num_layers):
    layers = [nn.Flatten(), nn.Linear(num_in, num_features), nn.GELU()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(num_features, num_features), nn.GELU()])
    layers.append(nn.Linear(num_features, num_out))
    return nn.Sequential(*layers)


def MobileNetV2(num_out):
    return torchvision.models.mobilenet_v2(num_classes=num_out)


def EfficientNetB0(num_out):
    return torchvision.models.efficientnet_b0(num_classes=num_out)


def EfficientNetB1(num_out):
    return torchvision.models.efficientnet_b1(num_classes=num_out)


def EfficientNetB2(num_out):
    return torchvision.models.efficientnet_b2(num_classes=num_out)


def EfficientNetB3(num_out):
    return torchvision.models.efficientnet_b3(num_classes=num_out)


def EfficientNetB4(num_out):
    return torchvision.models.efficientnet_b4(num_classes=num_out)


def EfficientNetB5(num_out):
    return torchvision.models.efficientnet_b5(num_classes=num_out)


def EfficientNetB6(num_out):
    return torchvision.models.efficientnet_b6(num_classes=num_out)


def EfficientNetB7(num_out):
    return torchvision.models.efficientnet_b7(num_classes=num_out)
