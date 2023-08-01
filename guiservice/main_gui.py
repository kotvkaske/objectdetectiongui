from tkinter_base import Window
from guiservice.utils import WebCam
import PIL
from PIL import Image, ImageTk
import cv2
import onnxruntime
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


cap = WebCam()
width, height = cap.GetCameraAttributes()
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
my_win = Window('FaceApp', f'{int(width - 100)}x{int(height - 100)}')
additive_win = my_win.create_child('second')
# my_model = SSD_Caffe((width, height), '../model_path/ssdcaffe')

# Deeplabm = DeepLabResnet()
# Deeplabm.load_state_dict(torch.load('../model_path/deeplab_weights.pt', map_location=torch.device(DEVICE)))
# Deeplabm = Deeplabm.to(DEVICE)
# Deeplabm.eval()
#
Deeplabm = ort_session = onnxruntime.InferenceSession("../dplmodel.onnx")


# segnet = SegNet()
# segnet.load_state_dict(torch.load('../model_path/segnet_weights.pt', map_location=torch.device(DEVICE)))
# segnet = segnet.to(DEVICE)
# segnet.eval()


def frame_to_img(frame_pic):
    cv2image = cv2.cvtColor(frame_pic, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    return imgtk


def video_preprocessing(vid=cap.camera, flag=additive_win.choice, extra_flag=additive_win.extra_choice):
    ret, image = vid.read()

    if flag.get() == 0:
        return image
    # elif flag.get() == 1:
    #     frame, myface = my_model.detect(image.copy())
    #     return frame
    elif flag.get() == 2:
        if extra_flag.get() == 0:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.tensor(image)/255)}
            ort_outs = ort_session.run(None, ort_inputs)
            return ort_outs[0]
            # return Deeplabm.foreground_extraction(image)

    #     elif extra_flag.get() == 1:
    #         return segnet.foreground_extraction(image)


def show_frame(window=my_win):
    frame = video_preprocessing()
    imgtk = frame_to_img(frame)
    window.lmain.imgtk = imgtk
    window.lmain.configure(image=imgtk)
    window.lmain.after(10, show_frame)


if __name__ == '__main__':
    show_frame()
    my_win.run()
