from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_classifier(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_resnet_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet(nn.Module):

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        channels = [base_channels * factor for factor in (1, 2, 4, 8, 16)]
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.encoder1 = DoubleConv(in_channels, channels[0])
        self.encoder2 = DoubleConv(channels[0], channels[1])
        self.encoder3 = DoubleConv(channels[1], channels[2])
        self.encoder4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(channels[4], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(channels[1], channels[0])
        self.output = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))
        bottleneck = self.bottleneck(self.pool(encoder4))

        decoder4 = self.decoder4(torch.cat((self.up4(bottleneck), encoder4), dim=1))
        decoder3 = self.decoder3(torch.cat((self.up3(decoder4), encoder3), dim=1))
        decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), dim=1))
        decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), dim=1))
        return self.output(decoder1)
