import numpy as np
import cv2
from BaseDetector import FaceDetector


class SSD(FaceDetector):
    def __init__(self, image_size: tuple, model_path: str = 'model_path', model_input_size: int = 300):
        self.model_path = model_path

        self.model = cv2.dnn.readNetFromCaffe(f'{self.model_path}/architecture.txt',
                                              f'{self.model_path}/weights.caffemodel')
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model_input_size = model_input_size
        self.image_size = image_size

    def predict_detections(self, image, display_confidence=True):
        blob = cv2.dnn.blobFromImage(image, 1.0, (self.model_input_size, self.model_input_size), crop=False)
        self.model.setInput(blob)
        detections = self.model.forward()
        return detections

