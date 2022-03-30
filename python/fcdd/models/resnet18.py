from typing import Optional

import torch.nn as nn
from fcdd.models.bases import FCDDNet, BaseNet
from torch.hub import load_state_dict_from_url

from torch import Tensor
from torch.hub import load_state_dict_from_url

class RESNET18(FCDDNet):
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        self.inplanes = 64

        self.conv1 = self._create_conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = self._create_maxpool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, receptive=True)
        # TRAIN
        self.layer2 = self._make_layer(128, 2, stride=2, receptive=True)
        # CUT
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(self.inplanes),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self._make_layer(64, 2),
        #     self._make_layer(128, 2, stride=2),
        #     self._make_layer(256, 2, stride=2),
        #     self._make_layer(512, 2, stride=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Linear(512, 1000)
        # )

        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth", progress = True)
        self.load_state_dict(state_dict)

        for child in list(self.children())[:-5]:
            for p in child.parameters():
                p.requires_grad = False

        self.conv_final = self._create_conv2d(128, 1, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    class BasicBlock(nn.Module):
        def __init__(self, outer, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, receptive=False) -> None:
            super().__init__()

            if receptive:
                self.conv1 = outer._create_conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = outer._create_conv2d(planes, planes, kernel_size=3, bias=False, padding=1)
                self.bn2 = nn.BatchNorm2d(planes)
            else:
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False, padding=1)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False, padding=1)
                self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, receptive=False) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if receptive:
                downsample = nn.Sequential(
                    self._create_conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        layers = []
        layers.append(self.BasicBlock(self, self.inplanes, planes, stride, downsample, receptive=receptive))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(self.BasicBlock(self, self.inplanes, planes, receptive=receptive))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, ad=True) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # TRAIN
        x = self.layer2(x)
        # CUT
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        if ad:
            x = self.conv_final(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

