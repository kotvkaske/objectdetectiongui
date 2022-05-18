from tkinter import *

from tkinter_base import Window
from model_base_ssd import *
from utils import WebCam
import PIL
from PIL import Image, ImageTk
import cv2

cap = WebCam()
width, height = cap.getcamattributes()



my_win = Window('FaceApp', f'{int(width - 100)}x{int(height - 100)}')
extra_funcs = my_win.create_child('second')
lmain = Label(my_win.window)
lmain.pack()
my_model = ModelDetection((width, height), 'model_path/ssdcaffe')


def frame_to_img(frame_pic):
    cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk


def show_frame():
    frame, myface = my_model.video_prediction(cap.camera)
    imgtk = frame_to_img(frame)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


show_frame()

my_win.run()
