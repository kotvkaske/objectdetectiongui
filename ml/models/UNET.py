import torch
import torch.nn.functional as F
import cv2
from torch import nn
from torchvision import models
import numpy as np
from abc import ABC, abstractmethod


def copy_crop_concat(x1, x2):  ##
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x


class convblock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel=3, padding=0, stride=1):
        super(convblock, self).__init__()
        self.first_conv = nn.Sequential(
            *[nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=padding, stride=stride),
              nn.BatchNorm2d(mid_channels),
              nn.BatchNorm2d(mid_channels),
              nn.ReLU(),
              nn.Conv2d(mid_channels, out_channels, kernel_size=kernel),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()])

    def forward(self, x):
        x = self.first_conv(x)
        return x


class UNET_custom(nn.Module):
    def __init__(self):
        super(UNET_custom, self).__init__()
        self.first_block = convblock(3, 64, 64, padding=2)
        self.first_max_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.second_block = convblock(64, 128, 128, padding=2)
        self.second_block_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.third_block = convblock(128, 256, 256, padding=2)
        self.third_block_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.forth_block = convblock(256, 512, 512, padding=2)
        #        self.forth_block_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.forth_deconvblock = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.forth_convblock = convblock(512, 256, 256, padding=2)

        self.third_deconvblock = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.third_convblock = convblock(256, 128, 128, padding=2)

        self.second_deconvblock = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.second_convblock = convblock(128, 64, 64, padding=2)

        self.outp_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.first_block(x)
        x1 = x.clone()
        x = self.first_max_pool(x)
        #        print(x.shape, 'спуск 1')
        x = self.second_block(x)
        x2 = x.clone()
        x = self.second_block_pool(x)
        #        print(x.shape, 'спуск 2')
        x = self.third_block(x)
        x3 = x.clone()
        x = self.third_block_pool(x)
        #        print(x.shape, 'спуск 3')

        x = self.forth_block(x)
        #        print(x.shape, 'спуск 4')

        x = self.forth_deconvblock(x)
        x = copy_crop_concat(x, x3)
        x = self.forth_convblock(x)

        # # print(x.shape, 'подъем 2')
        x = self.third_deconvblock(x)
        x = copy_crop_concat(x, x2)
        x = self.third_convblock(x)
        # # print(x.shape, 'подъем 3')

        x = self.second_deconvblock(x)
        x = copy_crop_concat(x, x1)
        x = self.second_convblock(x)
        x = self.outp_layer(x)
        return x

    def foreground_extraction(self, picture):
        w = picture.shape[1]
        h = picture.shape[0]
        pic = picture

        pic = cv2.resize(pic, dsize=(320, 240), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        pic = torch.tensor(pic) / 255
        result = self.model_custom(pic.to(DEVICE).unsqueeze(dim=0))['out'].squeeze(dim=0).detach().cpu()
        result = 255 * (result > 0.5).squeeze(dim=0).numpy().astype(np.uint8)
        pic = np.array(255 * pic).astype(np.uint8).transpose(1, 2, 0)
        pic[result == 0] = 0
        pic = cv2.resize(pic, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        return pic
