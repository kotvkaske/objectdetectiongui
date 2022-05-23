from tkinter import *
import torch
from tkinter_base import Window
from detection_models import *
from segmentation_models import *
from utils import WebCam
import PIL
from PIL import Image, ImageTk
import cv2
from torchvision import models
cap = WebCam()
width, height = cap.getcamattributes()
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
my_win = Window('FaceApp', f'{int(width - 100)}x{int(height - 100)}')
additive_win = my_win.create_child('second')
my_model = ModelDetection((width, height), 'model_path/ssdcaffe')

Deeplabm = DeepLabResnet()
Deeplabm.load_state_dict(torch.load('deeplab_weights.pt',map_location=torch.device(DEVICE)))
Deeplabm.eval()

segnet = SegNet()
segnet.load_state_dict(torch.load('segnet_finalx2aug.pth',map_location=torch.device(DEVICE)))
segnet.eval()



def frame_to_img(frame_pic):
    cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk


def video_preprocessing(vid=cap.camera,flag=additive_win.choice):
    ret, image = vid.read()

    if flag.get()==2:
        return segm_model.foreground_extraction(image)
    elif flag.get()==0:
        return image
    elif flag.get()==1:
        frame, myface = my_model.face_detection(ret,image.copy())
        return frame


def show_frame(window=my_win):
    frame = video_preprocessing()
    imgtk = frame_to_img(frame)
    window.lmain.imgtk = imgtk
    window.lmain.configure(image=imgtk)
    window.lmain.after(10, show_frame)

show_frame()
my_win.run()
