import PySimpleGUI as sg

def add_default_loading(funct) :
    def inner(*args,**kwargs) :
        waiting_layout = [
        [sg.Text("Running segmentation...")]
        ]
        window = sg.Window(
        title= 'small_fish',
        layout= waiting_layout,
        grab_anywhere= True,
        no_titlebar= True
    )

        window.read(timeout= 30, close= False)
        try :
            return funct(*args, **kwargs)
        finally :
            window.close()
    return inner

