import numpy as np
import cv2
from torch import nn
from imutils.video import FPS
from torchvision import models
import torch

class ModelDetection:
    def __init__(self, image_size: tuple, model_path='model_path', model_input_size=300):
        self.model_path = model_path
        self.model = cv2.dnn.readNetFromCaffe(f'{self.model_path}/architecture.txt',
                                              f'{self.model_path}/weights.caffemodel')
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model_input_size = model_input_size
        self.image_size = image_size

    def face_detextion(self, ret, image):
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), crop=False)
        w = self.image_size[0]
        h = self.image_size[1]
        self.model.setInput(blob)
        detections = self.model.forward()
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = detections[0, 0, i, 2]
            my_box = box.copy()
            if confidence > 0.7:
                my_box = image[startY:endY, startX:endX, :]
                try:
                    my_box = cv2.resize(my_box, dsize=(160, 195), interpolation=cv2.INTER_CUBIC)
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    # cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                except:
                    continue
        return image, my_box


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
        pic = torch.tensor(pic)/255
        result = self.model_custom(pic.unsqueeze(dim=0))['out'].squeeze(dim=0)
        result = 255*(result > 0.5).squeeze(dim=0).numpy().astype(np.uint8)
        pic = np.array(255*pic).astype(np.uint8).transpose(1,2,0)
        pic[result == 0] = 0
        pic = cv2.resize(pic, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        return pic