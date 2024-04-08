import PySimpleGUI as sg
import os
from inspect import getsourcefile
from os.path import abspath
from ..utils import check_parameter

sg.theme('DarkAmber')


def add_header(header_text, layout) :
    header = [[sg.Text('\n{0}'.format(header_text), size= (len(header_text),3), font= 'bold 15')]]
    return header + layout


def pad_right(string, length, pad_char) :
    if len(string) >= length : return string
    else : return string + pad_char* (length - len(string))
    

def parameters_layout(parameters:'list[str]' = [], header= None, default_values=None, size=5, opt=None) :

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
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), sg.InputText(size= size, key= parameter, default_text= value)] for parameter,value, option in zip(parameters,default_values, opt)
        ]
    else :
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), sg.InputText(size= size, key= parameter)] for parameter, option in zip(parameters, opt)
        ]
    if isinstance(header, str) :
        layout = add_header(header, layout)
    return layout

def tuple_layout(opt=None, default_dict={}, **tuples) :
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
        [sg.Text(pad_right(tup, max_size, ' '), text_color= 'green' if opt[option] else None)] + [sg.InputText(default_text=default_dict.setdefault('{0}_{1}'.format(tup,elmnt), elmnt),key= '{0}_{1}'.format(tup, elmnt), size= 5) for elmnt in tuples[tup]] for tup,option in zip(tuples,opt)
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

def combo_layout(values, key, header=None, read_only=True) :
    """
    drop-down list
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    layout = [
        sg.Combo(values, readonly=read_only, key=key)
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