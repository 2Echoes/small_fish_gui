import PySimpleGUI as sg
import os
from ..utils import check_parameter
import cellpose.models as models
from cellpose.core import use_gpu

sg.theme('DarkAmber')


def add_header(header_text, layout) :
    header = [[sg.Text('\n{0}'.format(header_text), size= (len(header_text),3), font= 'bold 15')]]
    return header + layout


def pad_right(string, length, pad_char) :
    if len(string) >= length : return string
    else : return string + pad_char* (length - len(string))
    

def parameters_layout(parameters:'list[str]' = [], unit=None, header= None, default_values=None, size=5, opt=None) :

    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header = (str, type(None)))
    for key in parameters : check_parameter(key = str)
    max_length = len(max(parameters, key=len))

    if type(opt) == type(None) :
        opt = [False] * len(parameters)
    else :
        if len(opt) != len(parameters) : raise ValueError("Parameters and opt must be of same length.")

    if isinstance(default_values, (list, tuple)) :
        if len(default_values) != len(parameters) : raise ValueError("if default values specified it must be of equal length as parameters.")
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), 
             sg.InputText(size= size, key= parameter, default_text= value)
             
             ] for parameter,value, option in zip(parameters,default_values, opt)
        ]
    else :
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), 
             sg.InputText(size= size, key= parameter)
             
             ] for parameter, option in zip(parameters, opt)
        ]
    
    if type(unit) == str :
        for line_id, line in enumerate(layout) :
            layout[line_id] += [sg.Text('{0}'.format(unit))]
    
    if isinstance(header, str) :
        layout = add_header(header, layout)
    return layout

def tuple_layout(opt=None, default_dict={}, unit:dict={}, **tuples) :
    """
    tuples example : voxel_size = ['z','y','x']; will ask a tuple with 3 element default to 'z', 'y' 'x'.
    """
    if type(tuples) == type(None) : return []
    if len(tuples.keys()) == 0 : return []

    if type(opt) != type(None) :
        if not isinstance(opt, dict) : raise TypeError("opt parameter should be either None or dict type.")
        if not opt.keys() == tuples.keys() : raise ValueError("If opt is passed it is expected to have same keys as tuples dict.")
    else : 
        opt = tuples.copy()
        for key in opt.keys() :
            opt[key] = False

    for tup in tuples : 
        if not isinstance(tuples[tup], (list,tuple)) : raise TypeError()

    max_size = len(max(tuples.keys(), key=len))
    
    layout = [
        [sg.Text(pad_right(tup, max_size, ' '), text_color= 'green' if opt[option] else None)] 
        + [sg.InputText(default_text=default_dict.setdefault('{0}_{1}'.format(tup,elmnt), elmnt),key= '{0}_{1}'.format(tup, elmnt), size= 5) for elmnt in tuples[tup]]
        + [sg.Text(unit.setdefault(tup,''))] 
        for tup,option, in zip(tuples,opt)
    ]

    return layout

def path_layout(keys= [],look_for_dir = False, header=None, preset=os.getcwd()) :
    """
    If not look for dir then looks for file.
    """
    if len(keys) == 0 : return []
    check_parameter(keys= list, header = (str, type(None)))
    for key in keys : check_parameter(key = str)
    if look_for_dir : Browse = sg.FolderBrowse
    else : Browse = sg.FileBrowse

    max_length = len(max(keys, key=len))
    layout = [
        [sg.Text(pad_right(name, max_length, ' ')), Browse(key= name, initial_folder= preset)] for name in keys
        ]
    if isinstance(header, str) :
        layout = add_header(header, layout=layout)
    return layout

def bool_layout(parameters= [], header=None, preset=None) :
    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header= (str, type(None)), preset=(type(None), list, tuple, bool))
    for key in parameters : check_parameter(key = str)
    if type(preset) == type(None) :
        preset = [False] * len(parameters)
    elif type(preset) == bool :
        preset = [preset] * len(parameters)
    else : 
        for key in preset : check_parameter(key = bool)



    max_length = len(max(parameters, key=len))
    layout = [
        [sg.Checkbox(pad_right(name, max_length, ' '), key= name, default=box_preset)] for name, box_preset in zip(parameters,preset)
    ]
    if isinstance(header, str) :
        layout = add_header(header, layout=layout)
    return layout

def combo_layout(values, key, header=None, read_only=True, default_value=None) :
    """
    drop-down list
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    if type(default_value) == type(None) :
        default_value = values[0]
    elif default_value not in values :
        default_value = values[0]
    layout = [
        sg.Combo(values, default_value=default_value, readonly=read_only, key=key)
    ]
    if isinstance(header, str) :
        layout = add_header(header, layout=layout)
    return layout

def radio_layout(values, header=None) :
    """
    Single choice buttons.
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    layout = [
        [sg.Radio(value, group_id= 0) for value in values]
    ]
    if isinstance(header, str) :
        layout = add_header(header, layout=layout)
    return layout

def _segmentation_layout(cytoplasm_model_preset= 'cyto2', nucleus_model_preset= 'nuclei', cytoplasm_channel_preset=0, nucleus_channel_preset=0, cyto_diameter_preset=30, nucleus_diameter_preset= 30, show_segmentation_preset= False, segment_only_nuclei_preset=False, saving_path_preset=os.getcwd(), filename_preset='cell_segmentation.png') :
    
    USE_GPU = use_gpu()

    models_list = models.get_user_models() + models.MODEL_NAMES
    if len(models_list) == 0 : models_list = ['no model found']
    
    #Header : GPU availabality
    layout = [[sg.Text("GPU is currently "), sg.Text('ON', text_color= 'green') if USE_GPU else sg.Text('OFF', text_color= 'red')]]
    
    #cytoplasm parameters
    layout += [add_header("Cell Segmentation", [sg.Text("Choose cellpose model for cytoplasm: \n")]),
              [combo_layout(models_list, key='cyto_model_name', default_value= cytoplasm_model_preset)]
                        ]
    layout += [parameters_layout(['cytoplasm channel'],default_values= [cytoplasm_channel_preset])]
    layout += [parameters_layout(['cytoplasm diameter'], unit= "px", default_values= [cyto_diameter_preset])]
    #Nucleus parameters
    layout += [
            add_header("Nucleus segmentation",[sg.Text("Choose cellpose model for nucleus: \n")]),
              combo_layout(models_list, key='nucleus_model_name', default_value= nucleus_model_preset)
                ]
    layout += [parameters_layout(['nucleus channel'], default_values= [nucleus_channel_preset])]
    layout += [parameters_layout([ 'nucleus diameter'],unit= "px", default_values= [nucleus_diameter_preset])]
    layout += [bool_layout(["Segment only nuclei"], preset=segment_only_nuclei_preset)]
    
    #Control plots
    layout += [bool_layout(['show segmentation'], header= 'Segmentation plots', preset= show_segmentation_preset)]
    layout += [path_layout(['saving path'], look_for_dir=True, preset=saving_path_preset)]
    layout += [parameters_layout(['filename'], default_values=[filename_preset], size= 25)]

    return layout