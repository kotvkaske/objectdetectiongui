import cv2


class WebCam():
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def GetCameraAttributes(self):
        return self.width, self.height



