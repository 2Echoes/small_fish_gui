import PySimpleGUI as sg
import pandas as pd
from .layout import path_layout, parameters_layout, bool_layout, tuple_layout
from ..interface import open_image, check_format, FormatError

def ask_input_parameters() :
    """
    Prompt user with interface allowing parameters setting for bigFish detection / deconvolution.
    
    Keys :
        - 'image path'
        - 'image'
        - '3D stack'
        - 'time stack'
        - 'multichannel'
        - 'Dense regions deconvolution'
        - 'Segmentation
        - 'Napari correction'
        - 'threshold'
        - 'time step'
        - 'channel to compute'
        - 'alpha'
        - 'beta'
        - 'gamma'
        - 'voxel_size_{(z,y,x)}'
        - 'spot_size{(z,y,x)}'
        - 'log_kernel_size{(z,y,x)}'
        - 'minimum_distance{(z,y,x)}'
    """
    
    values = {}

    image_input_values = input_image_prompt()
    values.update(image_input_values)

    pipeline_parameters_values = pipeline_parameters_promt(
        is_3D_stack=image_input_values['3D stack'], 
        is_time_stack=image_input_values['time stack'], 
        is_multichannel=image_input_values['multichannel'], 
        do_dense_region_deconvolution=image_input_values['Dense regions deconvolution'])
    values.update(pipeline_parameters_values)
    values['dim'] = 3 if values['3D stack'] else 2

    return values


def prompt(layout, add_ok_cancel=True) :
    if add_ok_cancel : layout += [[sg.Button('Ok'), sg.Button('Cancel')]]
    window = sg.Window('small fish', layout=layout, margins=(10,10))
    event, values = window.read()
    if event == None : 
        window.close()
        quit()
    elif event == 'Cancel' :
        window.close()
        return event,{}
    else : 
        window.close()
        return event, values


def input_image_prompt() :
    """
        Keys :
        - 'image path'
        - '3D stack'
        - 'time stack'
        - 'multichannel'
        - 'Dense regions deconvolution'

    Returns Values

    """
    layout_image_path = path_layout(['image path'], header= "Image")
    layout_image_path += bool_layout(['3D stack', 'time stack', 'multichannel'])
    layout_image_path += bool_layout(['Dense regions deconvolution', 'Segmentation', 'Napari correction'], header= "Pipeline settings")
    event, values = prompt(layout_image_path)
    im_path = values['image path']
    is_3D_stack = values['3D stack']
    is_time_stack = values['time stack']
    is_multichannel = values['multichannel']
    image = open_image(im_path)
    try : 
        check_format(image, is_3D_stack, is_time_stack, is_multichannel)
    except FormatError as error:
        sg.popup("Inconsistency between image format and options selected.\n Image shape : {0}".format(image.shape))
        

    values.update({'image' : image})

    return values


def output_image_prompt() :
    try :
        layout = path_layout(['folder'], look_for_dir= True, header= "Output parameters :")
        layout += parameters_layout(['filename'])
        layout += bool_layout(['Excel', 'Feather'])
        layout.append([sg.Button('Cancel')])

        event,values= prompt(layout)
    except Exception as error: 
        sg.popup('Error when saving files : {0}'.format(error))
        event = 'Cancel'

    if event == ('Cancel') : return False

    else : return values


def pipeline_parameters_promt(is_3D_stack, is_time_stack, is_multichannel, do_dense_region_deconvolution) :
    """
    keys :
        - 'threshold'
        - 'time step'
        - 'channel to compute'
        - 'alpha'
        - 'beta'
        - 'gamma'
        - 'voxel_size_{(z,y,x)}'
        - 'spot_size{(z,y,x)}'
        - 'log_kernel_size{(z,y,x)}'
        - 'minimum_distance{(z,y,x)}'

    Returns Values
        
    """
    if is_3D_stack : dim = 3
    else : dim = 2

    #Detection
    detection_parameters = ['threshold']
    if is_time_stack : detection_parameters += ['time step']
    if is_multichannel : detection_parameters += ['channel to compute']
    layout = parameters_layout(detection_parameters, header= 'Detection')
    if dim == 2 : tuple_shape = ('y','x')
    else : tuple_shape = ('z','y','x')
    layout += tuple_layout(voxel_size= tuple_shape, spot_size= tuple_shape, log_kernel_size= tuple_shape, minimum_distance= tuple_shape)

    #Deconvolution
    if do_dense_region_deconvolution :
        layout += parameters_layout(['alpha', 'beta', 'gamma'], default_values= [0.5, 1, 5], header= 'Dense regions deconvolution')
        layout += tuple_layout(deconvolution_kernel = tuple_shape)

    event, values = prompt(layout)
    return values


def post_analysis_prompt() :
    answer = events(['Save results','add_detection', 'colocalisation', 'open results in napari'])

    return answer


def events(event_list) :
    """
    Return event chose from user
    """
    
    layout = [
        [sg.Button(event) for event in event_list]
    ]

    event, values = prompt(layout, add_ok_cancel= False)
    return event


def _error_popup(error:Exception) :
    sg.popup('Error : ' + str(error))
    raise error

def _warning_popup(warning:str) :
    sg.popup('Warning : ' + warning)