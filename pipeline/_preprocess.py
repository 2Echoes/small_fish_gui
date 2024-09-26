import numpy as np
import os
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
        map = _auto_map_channels(is_3D_stack, is_time_stack, multichannel, image=image)
    except MappingError as e :
        sg.popup("Automatic dimension mapping went wrong. Please indicate dimensions positions in the array.")
        map = _ask_channel_map(image.shape, is_3D_stack, is_time_stack, multichannel, preset_map= e.get_map())

    else :
        map = _show_mapping(image.shape, map, is_3D_stack, is_time_stack, multichannel,)

    return map

def _auto_map_channels(is_3D_stack, is_time_stack, multichannel, image: np.ndarray=None, shape=None) :
    if type(shape) == type(None) :
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
    while True :
        relaunch = False
        save_preset = preset_map.copy()
        x = preset_map.setdefault('x',0)
        y = preset_map.setdefault('y',0)
        z = preset_map.setdefault('z',0)
        c = preset_map.setdefault('c',0)
        t = preset_map.setdefault('t',0)


        layout = [
            add_header("Dimensions mapping") + [sg.Text("Image shape : {0}".format(shape))]
        ]
        layout += [parameters_layout(['x','y'], default_values=[x,y])]
        if is_3D_stack : layout += [parameters_layout(['z'], default_values=[z])]
        if multichannel : layout += [parameters_layout(['c'], default_values=[c])]
        if is_time_stack : layout += [parameters_layout(['t'], default_values=[t])]

        event, preset_map = prompt_with_help(layout,help= 'mapping', add_scrollbar=False)
        if event == 'Cancel' : return save_preset

        #Check integrity
        channels_values = np.array(list(preset_map.values()), dtype= int)
        total_channels = len(preset_map)
        unique_channel = len(np.unique(channels_values))
        if total_channels != unique_channel :
            sg.popup("{0} channel(s) are not uniquely mapped.".format(total_channels - unique_channel))
            relaunch= True
        if not all(channels_values < len(shape)):
            sg.popup("Channels values out of range for image dimensions.\nPlease select dimensions from {0}".format(list(range(len(shape)))))
            relaunch= True
        if not relaunch : break

    return preset_map

def _show_mapping(shape, map, is_3D_stack, is_time_stack, multichannel) :
    while True : 
        layout = [
            [sg.Text("Image shape : {0}".format(shape))],
            [sg.Text('Dimensions mapping was set to :')],
            [sg.Text('x : {0} \ny : {1} \nz : {2} \nc : {3} \nt : {4}'.format(
                map['x'], map['y'], map.get('z'), map.get("c"), map.get('t')
            ))],
            [sg.Button('Change mapping')]
        ]

        event, values = prompt_with_help(layout, help='mapping', add_scrollbar=False)

        if event == 'Ok' :
            return map
        elif event == 'Change mapping':
            map = _ask_channel_map(shape, is_3D_stack, is_time_stack, multichannel, preset_map=map)
        elif event == 'Cancel' : 
            return None
        else : raise AssertionError('Unforseen event')


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
    int_list = ['threshold', 'channel_to_compute', 'channel to compute', 'min number of spots', 'cluster size','nucleus channel signal']
    float_list = ['alpha', 'beta', 'gamma', 'threshold penalty']

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

def check_integrity(
        values: dict, 
        do_dense_region_deconvolution,
        do_clustering, 
        multichannel,
        segmentation_done, 
        map, 
        shape
        ):
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

        if values['alpha'] > 1 or values['alpha'] < 0 :
            raise ParameterInputError("alpha must be set between 0 and 1.")

    if do_clustering :
        if not isinstance(values['min number of spots'], (int)) :
            raise ParameterInputError("Incorrect min spot number parameter.")
        if not isinstance(values['cluster size'], (int)) :
            raise ParameterInputError("Incorrect cluster size parameter.")

    #channel
    if multichannel :
        ch_len = shape[int(map['c'])]

        if type(segmentation_done) == type(None) :
            pass
        elif segmentation_done :
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

    #Spot extraction
    if not os.path.isdir(values['spots_extraction_folder']) and values['spots_extraction_folder'] != '':
        raise ParameterInputError("Incorrect spot extraction folder.")


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

def _check_segmentation_parameters(
        user_parameters,
        shape,
        is_multichannel,
) :

    available_channels = list(range(len(shape)))
    do_only_nuc = user_parameters['Segment only nuclei']
    cyto_model_name = user_parameters['cyto_model_name']
    cyto_size = user_parameters['cytoplasm diameter']
    cytoplasm_channel = user_parameters['cytoplasm channel']
    nucleus_model_name = user_parameters['nucleus_model_name']
    nucleus_size = user_parameters['nucleus diameter']
    nucleus_channel = user_parameters['nucleus channel']
   

    if type(cyto_model_name) != str  and not do_only_nuc:
        raise ParameterInputError('Invalid cytoplasm model name.')
    if cytoplasm_channel not in available_channels and not do_only_nuc and is_multichannel:
        raise ParameterInputError('For given input image please select channel in {0}\ncytoplasm channel : {1}'.format(available_channels, cytoplasm_channel))

    if type(cyto_size) not in [int, float] and not do_only_nuc:
        raise ParameterInputError("Incorrect cytoplasm size.")

    if type(nucleus_model_name) != str :
        raise ParameterInputError('Invalid nucleus model name.')
    
    if nucleus_channel not in available_channels and is_multichannel:
        raise ParameterInputError('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
    
    if type(nucleus_size) not in [int, float] :
        raise ParameterInputError("Incorrect nucleus size.")


def clean_unused_parameters_cache(user_parameters: dict) :
    """
    Clean unused parameters that were set to None in previous run.
    """
    parameters = ['alpha', 'beta', 'gamma', 'cluster size', 'min number of spots']
    for parameter in parameters :
        if parameter in user_parameters.keys() :
            if type(user_parameters[parameter]) == type(None) :
                del user_parameters[parameter]
    
    return user_parameters