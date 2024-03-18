import PySimpleGUI as sg


def _fake_help() :
    layout = [
        [sg.Text("Fake help window")],
        [sg.Button('Close')]
    ]
    window = sg.Window('small fish', layout=layout)
    event, values = window.read()

    if event == 'Close' :
        window.close()


def help_layout() :
    pass

def add_help_button(help_request) :
    pass

def _general_help() :
    pass

def _detection_help() :
    pass

def _segmentation_help() :
    layout =[
        [sg.Text("Segmentation is performed using Cellpose 2.0; this is published work that requires citation.\n Link to Cellpose Paper.")],
        [sg.Text("""Parameters : \n""")]
    ]

def _small_fish_help() :
    pass


layout = _segmentation_help()
window = sg.Window('test',layout=layout)
e,v=window.read()