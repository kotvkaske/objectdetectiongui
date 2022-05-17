from tkinter import *

from tkinter_base import Window
from model_base_ssd import *

import PIL
from PIL import Image,ImageTk
import cv2

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
my_win = Window('zdarova',f'{int(width-100)}x{int(height-100)}')
my_win.window.bind('<Escape>', lambda e: my_win.window.quit())
last_pic = my_win.create_child('second')
lmain = Label(my_win.window)
lmain.pack()
lnd = Label(last_pic.window)
lnd.pack()
my_model = ModelDetection((width,height),'model_path/ssdcaffe')


def frame_to_img(frame_pic):
    cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk

def show_frame():
    frame,myface = my_model.video_prediction(cap)
    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # img = PIL.Image.fromarray(cv2image)
    # imgtk = ImageTk.PhotoImage(image=img)
    imgtk = frame_to_img(frame)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    # imgtk_nd = frame_to_img(myface)
    # lnd.imgtk = imgtk_nd
    # lnd.configure(image=imgtk_nd)
    # lnd.after(10, show_frame)



show_frame()

my_win.run()