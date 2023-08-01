from abc import ABC, abstractmethod
import onnxruntime
import numpy as np
import cv2


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image, boxes, display_confidence=True):
        pass


class FaceDetector(BaseDetector):
    def __init__(self, path_to_model: str, type_of_model: str = 'ssd_caffe'):
        self.model_session = onnxruntime.InferenceSession(path_to_model)

    def detect(self, image, boxes, display_confidence=True):
        w, h = image.shape[0], image.shape[1]
        my_box = np.array([1, 1, 1, 1])
        for i in range(0, boxes.shape[2]):
            box = boxes[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = boxes[0, 0, i, 2]
            my_box = box.copy()
            if confidence > 0.7:
                my_box = image[startY:endY, startX:endX, :]
                try:
                    my_box = cv2.resize(my_box, dsize=(160, 195), interpolation=cv2.INTER_CUBIC)
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    if display_confidence:
                        cv2.putText(image, str(confidence), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (36, 255, 12), 2)
                except:
                    continue
        return image, my_box
