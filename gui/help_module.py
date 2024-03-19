import PySimpleGUI as sg
import os


def _fake_help() :
    layout = [
        [sg.Text("Fake help window")],
        [sg.Button('Close')]
    ]

    return layout

def ask_help(chapter= '') :
    
    if chapter == 'general' :
        help_l = _general_help()

    elif chapter == 'segmentation' :
        help_l= _segmentation_help()

    else : 
        help_l = _fake_help()

    window = sg.Window('Help (small fish)', layout=help_l, keep_on_top=True, auto_size_text=True)
    event, values = window.read(timeout= 0.1)

    if event == 'Close' :
        window.close()

def add_help_button(help_request) :
    pass

def _general_help() :

    im_path = os.path.dirname(__file__) + '/general_help_screenshot.png'

    help_text = """
    Pipeline settings :

        Dense regions deconvolution : (Recommanded for cluster computations) Detect dense and bright regions with potential clustered 
            spots and simulate a more realistic number of spots in these regions.
            See bigfish documentation : https://big-fish.readthedocs.io/en/stable/detection/dense.html
        
        Segmentation : Perform full cell segmentation in 2D (cytoplasm + nucleus) via cellpose.
            You can use your own retrained models or out of the box models from cellpose.

        Napari correct : (Not recommanded for time stack) 
            After each detection, opens a Napari viewer, enabling the user to visualise, add or remove spots.
    """

    layout = [
        [sg.Text("Welcome to small fish", font= 'bold 15')],
        [sg.Image(im_path)],
        [sg.Text(help_text, font = 'bold 10')]
    ]

    return layout

def _detection_help() :
    pass

def _segmentation_help() :

    cellpose1_quote = """Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
    Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106."""
    cellpose2_quote = """Pachitariu, M. & Stringer, C. (2022).
    Cellpose 2.0: how to train your own model. Nature methods, 1-8."""
    im_path = os.path.dirname(__file__) + '/segmentation_help_screenshot.png'

    layout =[
        [sg.Text("Segmentation is performed using Cellpose 2.0; this is published work that requires citation.\n")],
        [sg.Text(cellpose1_quote)],
        [sg.Text(cellpose2_quote)],
        [sg.Image(im_path)]
    ]

    return layout

def _small_fish_help() :
    pass