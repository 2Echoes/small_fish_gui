import PySimpleGUI as sg
import itertools
import threading
import time
import sys

def _default_loading(end_animation) :
    """
    Use a Thread to target this function, and set global parameter `end_animation` to True to finish.
    """
    while True :
        sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, time_between_frames=100)

        if end_animation :
            break

def add_default_loading(funct) :
    def inner(*args,**kwargs) :
        end_animation= False
        loading_thread = threading.Thread(target=_default_loading, args= (end_animation,), daemon=True)
        loading_thread.start()
        try :
            return funct(*args, **kwargs)
        finally :
            end_animation = True
    return inner

