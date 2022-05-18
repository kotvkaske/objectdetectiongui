import numpy as np
import cv2
from imutils.video import FPS


class ModelDetection:
    def __init__(self, image_size: tuple, model_path='model_path', model_input_size=300):
        self.model_path = model_path
        self.model = cv2.dnn.readNetFromCaffe(f'{self.model_path}/architecture.txt',
                                              f'{self.model_path}/weights.caffemodel')
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model_input_size = model_input_size
        self.image_size = image_size

    def video_prediction(self, vid):
        ret, image = vid.read()
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), crop=False)
        # fig = plt.figure(figsize=(14, 14))
        # rows = 3
        # cols = 2
        # axes = []
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
