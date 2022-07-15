import torch.nn as nn


def ConvBNAct(in_ch, out_ch, k_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k_size, stride, padding),
            ConvBNAct(in_ch, out_ch, k_size, stride, padding),
        )

    def forward(self, x):
        return self.layers(x) + x


def Classifier(num_classes):
    return nn.Sequential(
        ConvBNAct(3, 64, 3, 1, 1),
        ConvBNAct(64, 128, 3, 1, 1),
        nn.MaxPool2d(2, 2),
        ResBlock(128, 128, 3, 1, 1),
        ConvBNAct(128, 256, 3, 1, 1),
        nn.MaxPool2d(2, 2),
        ConvBNAct(256, 512, 3, 1, 1),
        nn.MaxPool2d(2, 2),
        ResBlock(512, 512, 3, 1, 1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
