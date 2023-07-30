import torch
import torch.nn.functional as F
import cv2
from torch import nn
from torchvision import models
import numpy as np
from abc import ABC, abstractmethod

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class HumanSegmentor(ABC):
    @abstractmethod
    def segment(self,image):
        pass


