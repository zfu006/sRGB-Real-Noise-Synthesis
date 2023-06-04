import torch
import torch.nn as nn
import numpy as np
import models.basicblocks as B

class Discriminator_96(nn.Module):
    """Discriminator with 96x96 input, refer to Kai Zhang, https://github.com/cszn/KAIR"""
    def __init__(self, in_nc=3, nc=64, act_mode='IL'):
        
        super(Discriminator_96, self).__init__()
        conv0 = B.conv(in_nc, nc, kernel_size=7, padding=3, mode='C')
        conv1 = B.conv(nc, nc, kernel_size=4, stride=2, mode='C'+act_mode)
        # 48, 64
        conv2 = B.conv(nc, nc*2, kernel_size=3, stride=1, mode='C'+act_mode)
        conv3 = B.conv(nc*2, nc*2, kernel_size=4, stride=2, mode='C'+act_mode)
        # 24, 128
        conv4 = B.conv(nc*2, nc*4, kernel_size=3, stride=1, mode='C'+act_mode)
        conv5 = B.conv(nc*4, nc*4, kernel_size=4, stride=2, mode='C'+act_mode)
        # 12, 256
        conv6 = B.conv(nc*4, nc*8, kernel_size=3, stride=1, mode='C'+act_mode)
        conv7 = B.conv(nc*8, nc*8, kernel_size=4, stride=2, mode='C'+act_mode)
        # 6, 512
        conv8 = B.conv(nc*8, nc*8, kernel_size=3, stride=1, mode='C'+act_mode)
        conv9 = B.conv(nc*8, nc*8, kernel_size=4, stride=2, mode='C'+act_mode)
        # 3, 512
        self.features = nn.Sequential(*[conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9])

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class Discriminator_32(nn.Module):
    def __init__(self, in_nc=3, nc=64, act_mode='IL'):
        super(Discriminator_32, self).__init__()
        # features
        # hxw, c
        # 32, 3
        conv0 = B.conv(in_nc, nc, kernel_size=7, padding=3, mode='C')
        conv1 = B.conv(nc, nc, kernel_size=4, stride=2, mode='C'+act_mode)
        # 16, 64
        conv2 = B.conv(nc, nc*2, kernel_size=3, stride=1, mode='C'+act_mode)
        conv3 = B.conv(nc*2, nc*2, kernel_size=4, stride=2, mode='C'+act_mode)
        # 8, 128
        conv4 = B.conv(nc*2, nc*4, kernel_size=3, stride=1, mode='C'+act_mode)
        conv5 = B.conv(nc*4, nc*4, kernel_size=4, stride=2, mode='C'+act_mode)
        # 4, 256

        self.features = nn.Sequential(*[conv0, conv1, conv2, conv3, conv4,
                                     conv5])

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x