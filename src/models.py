from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
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


# component of Unet
class DoubleConv(nn.Module):
    """Two convolution, batch-normalization, and ReLU blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# component of Unet
class Down(nn.Module):
    """Downscale with max pooling, then apply a double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# component of Unet
class Up(nn.Module):
    """Upscale and combine a decoder tensor with its skip connection."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.checkpointing = False

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self._run(self.inc, x)
        x2 = self._run(self.down1, x1)
        x3 = self._run(self.down2, x2)
        x4 = self._run(self.down3, x3)
        x5 = self._run(self.down4, x4)
        x = self._run(self.up1, x5, x4)
        x = self._run(self.up2, x, x3)
        x = self._run(self.up3, x, x2)
        x = self._run(self.up4, x, x1)
        return self._run(self.outc, x)

    def _run(self, module, *inputs):
        if self.checkpointing and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def use_checkpointing(self):
        self.checkpointing = True


class ResNet18Encoder(nn.Module):
    output_channels = (64, 64, 128, 256, 512)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stem = self.stem(x)
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4


class DecoderBlock(nn.Module):
    """upsample a feature map and fuse it with an encoder skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.conv(torch.cat((x, skip), dim=1))


# optional; didnt use in our study
class MultiTaskResNetUNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 4,
        segmentation_channels: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)

        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.segmentation_refine = DoubleConv(64, 32)
        self.segmentation_head = OutConv(32, segmentation_channels)
        self._encoder_trainable = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_size = x.shape[-2:]
        stem, layer1, layer2, layer3, layer4 = self.encoder(x)

        pooled = self.classifier_pool(layer4).flatten(1)
        class_logits = self.classifier(pooled)

        decoded = self.decoder4(layer4, layer3)
        decoded = self.decoder3(decoded, layer2)
        decoded = self.decoder2(decoded, layer1)
        decoded = self.decoder1(decoded, stem)
        decoded = F.interpolate(
            decoded,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        mask_logits = self.segmentation_head(self.segmentation_refine(decoded))
        return class_logits, mask_logits

    def train(self, mode: bool = True) -> MultiTaskResNetUNet:
        super().train(mode)
        if mode and not self._encoder_trainable:
            self.encoder.eval()
        return self

    def set_encoder_trainable(self, trainable: bool) -> None:
        self._encoder_trainable = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            self.encoder.eval()
