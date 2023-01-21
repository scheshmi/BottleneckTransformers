from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F


class MHSA(nn.Module):
    def __init__(self, n_dims: int, width: int, height: int, heads: int) -> None:
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        similarity = torch.matmul(q.permute(0, 1, 3, 2), k)

        relative_pos = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        query_pos = torch.matmul(relative_pos, q)

        energy = similarity + query_pos
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class Block(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            channels: int,
            downsample: nn.Sequential = None,
            stride: int = 1,
            mhsa: bool = False,
            heads: int = 4,
            shape: Tuple = None
    ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        if mhsa:
            self.conv2 = nn.Sequential()
            self.conv2.append(MHSA(channels, width=int(shape[0]), height=int(shape[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
        else:
            self.conv2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(
            channels,
            channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)
        return x


class BoTNet(nn.Module):
    def __init__(
            self,
            block: Block,
            layers: List[int],
            in_channels: int,
            num_classes: int,
            shape: Tuple,
            heads: int,
    ):
        super(BoTNet, self).__init__()

        self.channels = 64
        self.shape = shape

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.shape = [x // 4 for x in self.shape]

        self.layer1 = self._make_layer(block, layers[0], channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], channels=512, stride=1, mhsa=True, heads=heads)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(
            self,
            block: Block,
            n_block: int,
            channels: int,
            stride: int,
            mhsa: bool = False,
            heads: int = 4
    ):
        downsample = None
        layers = nn.Sequential()

        if stride != 1 or self.channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers.append(
            block(self.channels, channels, downsample, stride, mhsa, heads, self.shape)
        )

        if stride == 2:
            self.shape = [x // 2 for x in self.shape]

        self.channels = channels * block.expansion

        for i in range(n_block - 1):
            layers.append(block(self.channels, channels, mhsa=mhsa, heads=heads, shape=self.shape))

        return layers


def BoTNet50(num_classes=1000, input_shape=(224, 224), heads=4):
    return BoTNet(Block, [3, 4, 6, 3], in_channels=3, num_classes=num_classes, shape=input_shape, heads=heads)

