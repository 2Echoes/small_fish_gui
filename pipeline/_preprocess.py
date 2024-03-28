import numpy as np
import pandas as pd
import PySimpleGUI as sg
from ..gui import _error_popup, _warning_popup, parameters_layout, add_header, prompt

class ParameterInputError(Exception) :
    pass

def prepare_image_detection(map, image_stack) :
    """
    Generator yielding one image at a time for analysis. 
    Generator will have only one image if not time stack. 
    Re-arrange axis to [t,c,z,y,x]
    y,x are detected being the biggest dimensions
    c is detected being the smallest dimmension
    It can be hard to distinguish t and z : t will be the 1st element left in the list and z the 2nd. As in images formats t is usually the first dimension.

    Output
    ------
    Generator['np.ndarray']
        one frame per iteration [c,z,y,x]
    """
    
    image_stack = reorder_image_stack(map, image_stack) #
    if len(image_stack.shape) == 5 : #is time stack
        for image in image_stack :
            yield image
    else :
        yield image_stack

def reorder_image_stack(map, image_stack) :
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

class MappingError(ValueError) :
    def __init__(self, map ,*args: object) -> None:
        super().__init__(*args)
        self.map = map

    def get_map(self) :
        return self.map

def map_channels(image: np.ndarray, is_3D_stack, is_time_stack, multichannel) :
    
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

        event, map = prompt(layout)
        if event == 'Cancel' : quit()

        #Check integrity
        channels_values = np.array(list(map.values()), dtype= int)
        print(channels_values)
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
        [sg.Text('x : {0}, y : {1}, z : {2}, c : {3}, t : {4}'.format(
            map['x'], map['y'], map.get('z'), map.get("c"), map.get('t')
        ))],
        [sg.Button('Ok'), sg.Button('Change mapping')]
    ]

    event, values = prompt(layout, add_ok_cancel= False)

    if event == 'Ok' :
        return map
    elif event == 'Change mapping' :
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
    int_list = ['threshold', 'channel_to_compute', 'min number of spots', 'cluster size']
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

def check_integrity(values: dict, do_dense_region_deconvolution, is_time_stack, multichannel,map, shape):
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
    
    #time
    if is_time_stack :
        if type(values['time step']) == type(None) :
            raise ParameterInputError("Incorrect time_step.")
        elif values['time step'] == 0 :
            raise ParameterInputError("Incorrect time_step, must be > 0.")

    #channel
    if multichannel :
        try :
            ch = int(values['channel to compute'])
        except Exception :
            raise ParameterInputError("Incorrect channel to compute parameter.")
        ch_len = shape[int(map['c'])]
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

    print("new_shape : ", new_shape)

    return new_shape