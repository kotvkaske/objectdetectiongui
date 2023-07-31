from abc import ABC, abstractmethod


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image, display_confidence=True):
        pass
