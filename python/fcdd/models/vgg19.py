import os.path as pt

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fcdd.models.bases import FCDDNet, BaseNet
from torch.hub import load_state_dict_from_url
from torchvision import models


class VGG19(FCDDNet):
    # def __init__(self):
        # super(VGG16, self).__init__()
    def __init__(self, in_shape, **kwargs):
        super().__init__(in_shape, **kwargs)
        assert self.bias, 'VGG net is only supported with bias atm!'

        self.features = nn.Sequential(
            # 64
            self._create_conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            # 64
            self._create_conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            # M
            self._create_maxpool2d(2, 2),
            # 128
            self._create_conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            # 128
            self._create_conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            # M
            self._create_maxpool2d(2, 2),
            # 256
            self._create_conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            # 256
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            # 256
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            # 256
            self._create_conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            # M
            self._create_maxpool2d(2, 2),
            # 512
            self._create_conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # 512
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # 512
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # 512
            self._create_conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # CUT
            # M 
            nn.MaxPool2d(2, 2),
            # 512
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # 512
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # 512
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # 512
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            # M
            nn.MaxPool2d(2, 2)
        )

        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg19_bn-c79401a0.pth", progress = True)
        features_state_dict = {k[9:]: v for k, v in state_dict.items() if k.startswith('features')}
        self.features.load_state_dict(features_state_dict)

        self.features = self.features[:-14]

        for m in self.features[:30]:
            for p in m.parameters():
                p.requires_grad = False
        
        self.conv_final = self._create_conv2d(512, 1, 1)
        

    def forward(self, x, ad = True):
        x = self.features(x)
        if ad:
            x = self.conv_final(x)
        return x