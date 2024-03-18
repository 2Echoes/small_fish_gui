import PySimpleGUI as sg
from .layout import path_layout, parameters_layout, bool_layout, tuple_layout
from ..interface import open_image, check_format, FormatError
from .help import _fake_help


def prompt(layout, add_ok_cancel=True, timeout=None, timeout_key='TIMEOUT_KEY') :
    """
    Default event : 'Ok', 'Cancel'
    """
    if add_ok_cancel : layout += [[sg.Button('Ok'), sg.Button('Cancel')]]

    
    window = sg.Window('small fish', layout=layout, margins=(10,10))
    event, values = window.read(timeout=timeout, timeout_key=timeout_key)
    if event == None : 
        if ask_quit_small_fish() :
            window.close()
            quit()

    elif event == 'Cancel' :
        window.close()
        return event,{}
    else : 
        window.close()
        return event, values

def prompt_with_help(layout) :
    layout += [[sg.Button('Help')]]
    layout += [[sg.Button('Ok'), sg.Button('Cancel')]]
    
    window = sg.Window('small fish', layout=layout)
    while True :
        event, values = window.read()
            
        if event == None : 
            if ask_quit_small_fish() :
                window.close()
                quit()
            else : 
                print("Non")
                event, values = window.read()

        elif event == 'Cancel' :
            window.close()
            return event,{}
        elif event == 'Ok': 
            window.close()
            return event, values
        elif event == 'Help' :
            _fake_help()

def input_image_prompt(
        is_3D_stack_preset=False,
        time_stack_preset=False,
        multichannel_preset = False,
        do_dense_regions_deconvolution_preset= False,
        do_segmentation_preset= False,
        do_Napari_correction= False,
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
    layout_image_path += bool_layout(['Dense regions deconvolution', 'Segmentation', 'Napari correction'], preset= [do_dense_regions_deconvolution_preset, do_segmentation_preset, do_Napari_correction], header= "Pipeline settings")
    # event, values = prompt(layout_image_path)
    event, values = prompt_with_help(layout_image_path)

    if event == 'Cancel' :
        quit()

    im_path = values['image path']
    is_3D_stack = values['3D stack']
    is_time_stack = values['time stack']
    is_multichannel = values['multichannel']
    try :
        image = open_image(im_path)
        check_format(image, is_3D_stack, is_time_stack, is_multichannel)
        values.update({'image' : image})
    except FormatError as error:
        sg.popup("Inconsistency between image format and options selected.\n Image shape : {0}".format(image.shape))
    except OSError as error :
        sg.popup('Image format not supported.')


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


def detection_parameters_promt(is_3D_stack, is_time_stack, is_multichannel, do_dense_region_deconvolution) :
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