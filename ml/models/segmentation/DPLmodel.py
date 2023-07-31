import torch
import torch.nn.functional as F
import cv2
from torch import nn
from torchvision import models
import numpy as np
from abc import ABC, abstractmethod


class DeepLabResnet(nn.Module):
    """Архитектура deeplabv3_resnet50 для бинарной сегментации (0 - фон, 1 - человек),
    backbone - архитектура - Resnet50"""

    def __init__(self):
        super(DeepLabResnet, self).__init__()
        self.model_custom = models.segmentation.deeplabv3_resnet50(pretrained=True)
        for param in self.model_custom.parameters():
            param.requires_grad = False
        self.model_custom.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model_custom(x)['out']

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
