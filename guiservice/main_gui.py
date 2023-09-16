from guiservice.tkinter_base import Window
from guiservice.utils import WebCam
import PIL
from PIL import Image, ImageTk
from tkinter import IntVar
import cv2
import onnxruntime
import torch
import numpy as np
from ml.models.segmentation.BaseSegmentor import *


def frame_to_img(frame_pic):
    cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk


def video_preprocessing(vid: cv2.cv2.VideoCapture, flag: IntVar, extra_flag: IntVar):
    ret, image = vid.read()

    if flag.get() == 0:
        return image
    # elif flag.get() == 1:
    #     frame, myface = my_model.detect(image.copy())
    #     return frame
    # elif flag.get() == 2:
    #     if extra_flag.get() == 0:
    #         return model.segment(image)
    #         # return Deeplabm.foreground_extraction(image)
    #
    # #     elif extra_flag.get() == 1:
    # #         return segnet.foreground_extraction(image)


def show_frame(window: Window):
    frame = video_preprocessing()
    imgtk = frame_to_img(frame)
    window.lmain.imgtk = imgtk
    window.lmain.configure(image=imgtk)
    window.lmain.after(10, show_frame)
