from guiservice.tkinter_base import Window
from guiservice.utils import WebCam
from guiservice.main_gui import *
import PIL
from PIL import Image, ImageTk
import cv2
import onnxruntime
import torch
import numpy as np
from ml.models.segmentation.BaseSegmentor import *
import configparser

cfg = configparser.ConfigParser()
cfg.read('config.ini')

cap = WebCam()
width, height = cap.GetCameraAttributes()
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
my_win = Window('FaceApp', f'{int(width - 100)}x{int(height - 100)}')
additive_win = my_win.create_child('Model Selection')
# model = BodySegmentor('../dplmodel.onnx')

if __name__ == '__main__':
    show_frame(my_win, cap.camera)
    my_win.run()
