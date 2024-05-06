import numpy as np
import pandas as pd
import PySimpleGUI as sg
from ..gui import _error_popup, _warning_popup, parameters_layout, add_header, prompt, prompt_with_help

class ParameterInputError(Exception) :
    """
    Raised when user inputs an incorrect parameter.
    """
    pass

class MappingError(ValueError) :
    """
    Raised when user inputs an incorrect image mapping.
    """
    def __init__(self, map ,*args: object) -> None:
        super().__init__(*args)
        self.map = map

    def get_map(self) :
        return self.map

def prepare_image_detection(map, user_parameters) :
    """
    Return monochannel image for ready for spot detection; 
    if image is already monochannel, nothing happens.
    else : image is the image on which detection is performed, other_image are the other layer to show in Napari Viewer.
    """
    image = reorder_image_stack(map, user_parameters)
    assert len(image.shape) != 5 , "Time stack not supported, should never be True"
    
    if user_parameters['multichannel'] :
        channel_to_compute = user_parameters['channel to compute']
        other_image = image.copy()
        other_image = np.delete(other_image, channel_to_compute, axis=0)
        other_image = [layer for layer in other_image]
        image: np.ndarray = image[channel_to_compute]

    else :
        other_image = []

    return image, other_image

def reorder_image_stack(map, user_parameters) :
    image_stack = user_parameters['image']
    x = (int(map['x']),)
    y = (int(map['y']),)
    z = (int(map['z']),) if type(map.get('z')) != type(None) else ()
    c = (int(map['c']),) if type(map.get('c')) != type(None) else ()
    t = (int(map['t']),) if type(map.get('t')) != type(None) else ()

    source = t+c+z+y+x

    image_stack = np.moveaxis(
        image_stack,
        source= source,
        destination= tuple(range(len(source)))
    )

    return image_stack

def map_channels(user_parameters) :
    
    image = user_parameters['image']
    is_3D_stack = user_parameters['3D stack']
    is_time_stack = user_parameters['time stack']
    multichannel = user_parameters['multichannel']

    try : 
        map = _auto_map_channels(image, is_3D_stack, is_time_stack, multichannel)
    except MappingError as e :
        sg.popup("Automatic dimension mapping went wrong. Please indicate manually dimensions positions in the array.")
        map = _ask_channel_map(image.shape, is_3D_stack, is_time_stack, multichannel, preset_map= e.get_map())

    else :
        map = _show_mapping(image.shape, map, is_3D_stack, is_time_stack, multichannel,)

    return map

def _auto_map_channels(image: np.ndarray, is_3D_stack, is_time_stack, multichannel) :
    shape = image.shape
    reducing_list = list(shape)

    #Set the biggest dimension to y
    y_val = max(reducing_list)
    y_idx = shape.index(y_val)
    map = {'y' : y_idx}

    #2nd biggest set to x
    reducing_list[y_idx] = -1
    x_val = max(reducing_list)
    x_idx = reducing_list.index(x_val)
    reducing_list[y_idx] = y_val

    map['x'] = x_idx
    reducing_list.remove(y_val)
    reducing_list.remove(x_val)

    #smaller value set to c
    if multichannel :
        c_val = min(reducing_list)
        c_idx = shape.index(c_val)
        map['c'] = c_idx
        reducing_list.remove(c_val)

    if is_time_stack :
        t_val = reducing_list[0]
        t_idx = shape.index(t_val)
        map['t'] = t_idx
        reducing_list.remove(t_val)
    
    if is_3D_stack :
        z_val = reducing_list[0]
        z_idx = shape.index(z_val)
        map['z'] = z_idx

    total_channels = len(map)
    unique_channel = len(np.unique(list(map.values())))

    if total_channels != unique_channel : raise MappingError(map,"{0} channel(s) are not uniquely mapped.".format(total_channels - unique_channel))

    return map

def _ask_channel_map(shape, is_3D_stack, is_time_stack, multichannel, preset_map: dict= {}) :
    map = preset_map
    while True :
        relaunch = False
        x = map.setdefault('x',0)
        y = map.setdefault('y',0)
        z = map.setdefault('z',0)
        c = map.setdefault('c',0)
        t = map.setdefault('t',0)

        layout = [
            add_header("Dimensions mapping", [sg.Text("Image shape : {0}".format(shape))])
        ]
        layout += [parameters_layout(['x','y'], default_values=[x,y])]
        if is_3D_stack : layout += [parameters_layout(['z'], default_values=[z])]
        if multichannel : layout += [parameters_layout(['c'], default_values=[c])]
        if is_time_stack : layout += [parameters_layout(['t'], default_values=[t])]

        event, map = prompt_with_help(layout,help= 'mapping')
        if event == 'Cancel' : quit()

        #Check integrity
        channels_values = np.array(list(map.values()), dtype= int)
        total_channels = len(map)
        unique_channel = len(np.unique(channels_values))
        if total_channels != unique_channel :
            sg.popup("{0} channel(s) are not uniquely mapped.".format(total_channels - unique_channel))
            relaunch= True
        if not all(channels_values < len(shape)):
            sg.popup("Channels values out of range for image dimensions.\nPlease select dimensions from {0}".format(list(range(len(shape)))))
            relaunch= True
        if not relaunch : break

    return map

def _show_mapping(shape, map, is_3D_stack, is_time_stack, multichannel) :
    layout = [
        [sg.Text("Image shape : {0}".format(shape))],
        [sg.Text('Dimensions mapping was set to :')],
        [sg.Text('x : {0} \ny : {1} \nz : {2} \nc : {3} \nt : {4}'.format(
            map['x'], map['y'], map.get('z'), map.get("c"), map.get('t')
        ))],
        [sg.Button('Change mapping')]
    ]

    event, values = prompt_with_help(layout, help='mapping')

    if event == 'Ok' :
        return map
    elif event == 'Change mapping' or event == 'Cancel':
        map = _ask_channel_map(shape, is_3D_stack, is_time_stack, multichannel, preset_map=map)
    else : raise AssertionError('Unforseen event')

    return map

def convert_parameters_types(values:dict) :
    """
    Convert parameters from `ask_input_parameters` from strings to float, int or tuple type.
    """

    #Tuples
    tuples_list = ['voxel_size', 'spot_size', 'log_kernel_size', 'minimum_distance', 'deconvolution_kernel']
    dim = values['dim']
    if dim == 3 : dim_tuple = ('z', 'y', 'x')
    else : dim_tuple = ('y', 'x')

    for tuple_parameter in tuples_list :
        try :
            tuple_values = tuple([float(values.get(tuple_parameter + '_{0}'.format(dimension))) for dimension in dim_tuple])
        except Exception : #execption when str cannot be converted to float or no parameter was given.
            values[tuple_parameter] = None
        else : values[tuple_parameter] = tuple_values

    #Parameters
    int_list = ['threshold', 'channel_to_compute', 'min number of spots', 'cluster size','nucleus channel signal']
    float_list = ['time_step', 'alpha', 'beta', 'gamma', 'threshold penalty']

    for parameter in int_list :
        try :
            parameter_value = int(values[parameter])
        except Exception :
            values[parameter] = None
        else : values[parameter] = parameter_value

    for parameter in float_list :
        try :
            parameter_value = float(values[parameter])
        except Exception :
            values[parameter] = None
        else : values[parameter] = parameter_value

    return values

def check_integrity(values: dict, do_dense_region_deconvolution, multichannel,segmentation_done, map, shape):
    """
    Checks that parameters given in input by user are fit to be used for bigfish detection.
    """

    #voxel_size
    if type(values['voxel_size']) == type(None) : raise ParameterInputError('Incorrect voxel size parameter.')
    
    #detection integrity :
    if not isinstance(values['spot_size'], (tuple, list)) and not(isinstance(values['minimum_distance'], (tuple, list)) and isinstance(values['log_kernel_size'], (tuple, list))) :
       raise ParameterInputError("Either minimum_distance and 'log_kernel_size' must be correctly set\n OR 'spot_size' must be correctly set.")
    
    #Deconvolution integrity
    if do_dense_region_deconvolution :
        if not isinstance(values['alpha'], (float, int)) or not isinstance(values['beta'], (float, int)) :
            raise ParameterInputError("Incorrect alpha or beta parameters.")
        if type(values['gamma']) == type(None) and not isinstance(values['deconvolution_kernel'], (list, tuple)):
            _warning_popup('No gamma found; image will not be denoised before deconvolution.')
            values['gamma'] = 0

    #channel
    if multichannel :
        ch_len = shape[int(map['c'])]
        if segmentation_done :
            try : nuc_signal_ch = int(values['nucleus channel signal'])
            except Exception :
                raise ParameterInputError("Incorrect channel for nucleus signal measure.")
            if nuc_signal_ch > ch_len :
                raise ParameterInputError("Nucleus signal channel is out of range for image.\nPlease select from {0}".format(list(range(ch_len))))
            values['nucleus channel signal'] = nuc_signal_ch

        try :
            ch = int(values['channel to compute'])
        except Exception :
            raise ParameterInputError("Incorrect channel to compute parameter.")
        if ch >= ch_len :
            raise ParameterInputError("Channel to compute is out of range for image.\nPlease select from {0}".format(list(range(ch_len))))
        values['channel to compute'] = ch

    return values


def reorder_shape(shape, map) :
    x = [int(map['x']),]
    y = [int(map['y']),]
    z = [int(map['z']),] if type(map.get('z')) != type(None) else []
    c = [int(map['c']),] if type(map.get('c')) != type(None) else []
    t = [int(map['t']),] if type(map.get('t')) != type(None) else []

    source = t + c + z + y + x

    new_shape = tuple(
        np.array(shape)[source]
    )

    return new_shape