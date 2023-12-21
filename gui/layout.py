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
    

def parameters_layout(parameters:'list[str]' = [], header= None, default_values=None) :

    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header = (str, type(None)))
    for key in parameters : check_parameter(key = str)
    max_length = len(max(parameters, key=len))
    if isinstance(default_values, (list, tuple)) :
        if len(default_values) != len(parameters) : raise ValueError("if default values specified it must be of equal length as parameters.")
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' '))), sg.InputText(size= 5, key= parameter, default_text= value)] for parameter,value in zip(parameters,default_values)
        ]
    else :
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' '))), sg.InputText(size= 5, key= parameter)] for parameter in parameters
        ]
    if isinstance(header, str) :
        layout = add_header(header, layout)
    return layout

def tuple_layout(**tuples) :
    """
    tuples example : voxel_size = ['z','y','x']; will ask a tuple with 3 element default to 'z', 'y' 'x'.
    """
    if type(tuples) == type(None) : return []
    if len(tuples.keys()) == 0 : return []
    for tup in tuples : 
        if not isinstance(tuples[tup], (list,tuple)) : raise TypeError()

    max_size = len(max(tuples.keys(), key=len))
    
    layout = [
        [sg.Text(pad_right(tup, max_size, ' '))] + [sg.InputText(elmnt,key= '{0}_{1}'.format(tup, elmnt), size= 5) for elmnt in tuples[tup]] for tup in tuples
    ]

    return layout

def path_layout(keys= [],look_for_dir = False, header=None) :
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
        [sg.Text(pad_right(name, max_length, ' ')), Browse(key= name, initial_folder= os.getcwd())] for name in keys
        ]
    if isinstance(header, str) :
        layout = add_header(header, layout=layout)
    return layout

def bool_layout(parameters= [], header=None) :
    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header= (str, type(None)))
    for key in parameters : check_parameter(key = str)

    max_length = len(max(parameters, key=len))
    layout = [
        [sg.Checkbox(pad_right(name, max_length, ' '), key= name)] for name in parameters
    ]
    if isinstance(header, str) :
        layout = add_header(header, layout=layout)
    return layout