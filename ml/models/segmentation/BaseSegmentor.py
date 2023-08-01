from abc import ABC, abstractmethod
import torch
import cv2
import onnxruntime
import numpy as np
from ..utils import to_numpy


class BaseSegmentor(ABC):
    @abstractmethod
    def segment(self, image, display_confidence=True):
        pass


class BodySegmentor(BaseSegmentor):
    def __init__(self, path_to_model: str):
        self.model_session = onnxruntime.InferenceSession(path_to_model)

    def segment(self, image, display_confidence=True):
        w,h = image.shape[0], image.shape[1]
        resized_img = cv2.resize(image, dsize=(240, 320), interpolation=cv2.INTER_CUBIC)
        resized_img = np.transpose(resized_img, (2, 0, 1)) / 255
        ort_inputs = {
            self.model_session.get_inputs()[0].name: to_numpy(torch.tensor(resized_img).unsqueeze(dim=0).float())}
        ort_outs = self.model_session.run(None, ort_inputs)[0]
        result = 255 * (ort_outs > 0.5).squeeze().astype(np.uint8)
        result = cv2.resize(result, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        image[result == 0] = 0

        return image
