from tkinter import *


class Window:
    def __init__(self, title_str: str, geometry_size='200x500'):
        self.window = Tk()
        self.window.title(title_str)
        self.window.geometry(geometry_size)
        self.window.bind('<Escape>', lambda e: self.window.quit())

    def run(self):
        self.window.mainloop()

    def create_child(self, title_str: str, geometry_size='200x500'):
        return ChildWindow(self.window, title_str, geometry_size)


class ChildWindow:
    def __init__(self, parent, title_str: str, geometry_size='200x500'):
        self.window = Toplevel(parent)
        self.window.title(title_str)
        self.window.geometry(geometry_size)
