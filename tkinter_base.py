from tkinter import *


class Window:
    def __init__(self, title_str: str, geometry_size='200x500'):
        self.window = Tk()
        self.window.title(title_str)
        self.window.geometry(geometry_size)
        self.window.bind('<Escape>', lambda e: self.window.quit())
        self.lmain = Label(self.window)

    def run(self):
        self.draw_vidgets()
        self.window.mainloop()

    def draw_vidgets(self):
        self.lmain.pack()

    def create_child(self, title_str: str, geometry_size='200x500'):
        return ChildWindow(self.window, title_str, geometry_size)


class ChildWindow:
    def __init__(self, parent, title_str: str, geometry_size='200x500'):
        self.window = Toplevel(parent)
        self.window.title(title_str)
        self.window.geometry(geometry_size)
        self.label = Label(self.window)
        self.choice = IntVar(value=0)
        self.extra_choice = IntVar(value=0)
        self.draw_widgets()

    def draw_widgets(self):
        Radiobutton(self.window, text='Empty', variable=self.choice, value=0).pack()
        Radiobutton(self.window, text='Detection', variable=self.choice, value=1).pack()
        seg_button = Radiobutton(self.window, text='Segmentation', variable=self.choice, value=2)
        seg_button.pack()
        men = Menubutton(self.window,text='Model_Type')
        men.menu = Menu(men)
        men['menu'] = men.menu
        men.menu.add_radiobutton(label = 'SegNet',variable= self.extra_choice,value=0)
        men.menu.add_radiobutton(label='DeepLabv3Resnet50', variable=self.extra_choice, value=1)
        men.pack()


