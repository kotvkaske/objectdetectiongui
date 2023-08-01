from abc import ABC, abstractmethod


class BodySegmentor(ABC):
    @abstractmethod
    def segment(self, image, display_confidence=True):
        pass
