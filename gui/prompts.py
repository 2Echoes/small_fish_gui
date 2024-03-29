import PySimpleGUI as sg
import pandas as pd
import numpy as np
from .layout import path_layout, parameters_layout, bool_layout, tuple_layout
from ..interface import open_image, check_format, FormatError
from .help_module import ask_help

def prompt(layout, add_ok_cancel=True, timeout=None, timeout_key='TIMEOUT_KEY') :
    """
    Default event : 'Ok', 'Cancel'
    """
    if add_ok_cancel : layout += [[sg.Button('Ok'), sg.Button('Cancel')]]

    
    window = sg.Window('small fish', layout=layout, margins=(10,10))
    event, values = window.read(timeout=timeout, timeout_key=timeout_key)
    if event == None : 
        window.close()
        quit()

    elif event == 'Cancel' :
        window.close()
        return event,{}
    else : 
        window.close()
        return event, values

def prompt_with_help(layout, help =None) :
    layout += [[]]
    layout += [[sg.Button('Help')]]
    layout += [[sg.Button('Ok'), sg.Button('Cancel')]]
    
    window = sg.Window('small fish', layout=layout)
    while True :
        event, values = window.read()
        if event == None :
            window.close()
            quit()

        elif event == 'Ok': 
            window.close()
            return event, values
        elif event == 'Help' :
            ask_help(chapter= help)
        
        else:
            window.close()
            return event,{}

def input_image_prompt(
        is_3D_stack_preset=False,
        time_stack_preset=False,
        multichannel_preset = False,
        do_dense_regions_deconvolution_preset= False,
        do_clustering_preset = False,
        do_segmentation_preset= False,
        do_Napari_correction= False,
        ask_for_segmentation= True
) :
    """
        Keys :
        - 'image path'
        - '3D stack'
        - 'time stack'
        - 'multichannel'
        - 'Dense regions deconvolution'
        - 'Segmentation'
        - 'Napari correction'

    Returns Values

    """
    layout_image_path = path_layout(['image path'], header= "Image")
    layout_image_path += bool_layout(['3D stack', 'time stack', 'multichannel'], preset= [is_3D_stack_preset, time_stack_preset, multichannel_preset])
    
    if ask_for_segmentation : 
        layout_image_path += bool_layout(['Dense regions deconvolution', 'Cluster computation', 'Segmentation', 'Napari correction'], preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_segmentation_preset, do_Napari_correction], header= "Pipeline settings")
    else : 
        layout_image_path += bool_layout(['Dense regions deconvolution', 'Cluster computation', 'Napari correction'], preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_Napari_correction], header= "Pipeline settings")
    
    event, values = prompt_with_help(layout_image_path, help= 'general')

    if event == 'Cancel' :
        return None

    im_path = values['image path']
    is_3D_stack = values['3D stack']
    is_time_stack = values['time stack']
    is_multichannel = values['multichannel']
    if not ask_for_segmentation : values['Segmentation'] = False

    if is_time_stack :
        sg.popup("Sorry time stack images are not yet supported.")
        return values
    
    try :
        image = open_image(im_path)
        check_format(image, is_3D_stack, is_time_stack, is_multichannel)
        values.update({'image' : image})
    except FormatError as error:
        sg.popup("Inconsistency between image format and options selected.\n Image shape : {0}".format(image.shape))
    except OSError as error :
        sg.popup('Image format not supported.')
    except ValueError as error :
        sg.popup('Image format not supported.')


    return values

def output_image_prompt() :
    try :
        layout = path_layout(['folder'], look_for_dir= True, header= "Output parameters :")
        layout += parameters_layout(['filename'], size=25)
        layout += bool_layout(['Excel', 'Feather'])
        layout.append([sg.Button('Cancel')])

        event,values= prompt(layout)
    except Exception as error: 
        sg.popup('Error when saving files : {0}'.format(error))
        event = 'Cancel'

    if event == ('Cancel') : return None

    else : return values

def detection_parameters_promt(is_3D_stack, is_time_stack, is_multichannel, do_dense_region_deconvolution, do_clustering) :
    """

    keys :
        - 'threshold'
        - 'threshold penalty
        - 'time step'
        - 'channel to compute'
        - 'alpha'
        - 'beta'
        - 'gamma'
        - 'voxel_size_{(z,y,x)}'
        - 'spot_size{(z,y,x)}'
        - 'log_kernel_size{(z,y,x)}'
        - 'minimum_distance{(z,y,x)}'
        - 'cluster size'
        - 'min number of spots'

    Returns Values
        
    """
    if is_3D_stack : dim = 3
    else : dim = 2

    #Detection
    detection_parameters = ['threshold', 'threshold penalty']
    opt= [True, True]
    if is_time_stack : 
        detection_parameters += ['time step']
        opt += [False]
    if is_multichannel : 
        detection_parameters += ['channel to compute']
        opt += [False]
    layout = [[sg.Text("Green parameters", text_color= 'green'), sg.Text(" are optional parameters.")]]
    layout += parameters_layout(detection_parameters, header= 'Detection', opt=opt)
    
    if dim == 2 : tuple_shape = ('y','x')
    else : tuple_shape = ('z','y','x')
    opt = {'voxel_size' : False, 'spot_size' : False, 'log_kernel_size' : True, 'minimum_distance' : True}
    layout += tuple_layout(opt=opt, voxel_size= tuple_shape, spot_size= tuple_shape, log_kernel_size= tuple_shape, minimum_distance= tuple_shape)

    #Deconvolution
    if do_dense_region_deconvolution :
        layout += parameters_layout(['alpha', 'beta', 'gamma'], default_values= [0.5, 1, 5], header= 'Dense regions deconvolution')
        layout += tuple_layout(opt= {"deconvolution_kernel" : True}, deconvolution_kernel = tuple_shape)
    
    #Clustering
    if do_clustering :
        layout += parameters_layout(['cluster size', 'min number of spots'], default_values=[400, 5])

    event, values = prompt_with_help(layout, help='detection')
    if event == 'Cancel' : return None
    if is_3D_stack : values['dim'] = 3
    else : values['dim'] = 2
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

def ask_cancel_segmentation() :
    layout = [
        [sg.Text("Cancel segmentation ?")],
        [sg.Button('Yes'), sg.Button('No')]
    ]

    event, values = prompt(layout, add_ok_cancel= False)

    return event == 'Yes'

def ask_quit_small_fish() :
    layout = [
        [sg.Text("Quit small fish ?")],
        [sg.Button('Yes'), sg.Button('No')]
    ]

    event, values = prompt(layout, add_ok_cancel= False)

    return event == 'Yes'

def _error_popup(error:Exception) :
    sg.popup('Error : ' + str(error))
    raise error

def _warning_popup(warning:str) :
    sg.popup('Warning : ' + warning)

def _sumup_df(results: pd.DataFrame) :

    res = results.loc[:,['acquisition_id', 'spot_number', 'cell_number', 'filename', 'channel to compute', 'time']]

    return res

def hub_prompt(fov_results_list:list, do_segmentation=False) :

    sumup_df = _sumup_df(fov_results_list)
    if do_segmentation :
        segmentation_object = sg.Text('Segmentation was performed', font='8', text_color= 'green')
    else :
        segmentation_object = sg.Text('Segmentation was not performed', font='8', text_color= 'red')

    layout = [
        [sg.Text('RESULTS', font= 'bold 13')],
        [sg.Table(values= list(sumup_df.values), headings= list(sumup_df.columns), row_height=20, num_rows= 5, vertical_scroll_only=False, key= "result_table"), segmentation_object],
        [sg.Button('Add detection'), sg.Button('Compute colocalisation'), sg.Button('Save results')]
    ]

    window = sg.Window('small fish', layout= layout, margins= (10,10))

    while True : 
        event, values = window.read()
        if event == None : quit()
        elif event == 'Help' : pass
        else : 
            window.close()
            return event, values

def coloc_prompt() :
    layout = [
        [parameters_layout(['colocalisation distance'], header= 'Colocalisation', default_values= 0)]
    ]

    event, values = prompt_with_help(layout)

    if event == 'Ok' :
        return values['colocalisation distance']
    else : return False