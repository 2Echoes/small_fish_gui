from ..gui import _error_popup, _warning_popup

class ParameterInputError(Exception) :
    pass


def prepare_image(image_stack, is_3D_stack, is_time_stack, multichannel, channel_to_compute=0) :
    """
    Generator yielding one image at a time for analysis.
    """
    
    if multichannel and is_time_stack :
        image = image_stack[:,channel_to_compute,:]
    elif multichannel :
        image = image_stack[channel_to_compute,:]
        print(image.shape)
    if not is_time_stack : image_stack = [image_stack]
    for image in image_stack : 
        if is_3D_stack : assert image.ndim >= 3
        else : assert image.ndim == 2
        yield image




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
    int_list = ['threshold', 'channel_to_compute']
    float_list = ['time_step', 'alpha', 'beta', 'gamma']

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


def check_integrity(values: dict):
    """
    Checks that parameters given in input by user are fit to be used for bigfish detection.
    """

    #voxel_size
    if type(values['voxel_size']) == type(None) : _error_popup(ParameterInputError('Incorrect voxel size parameter.'))
    
    #detection integrity :
    if not isinstance(values['spot_size'], (tuple, list)) and not(isinstance(values['minimum_distance'], (tuple, list)) and isinstance(values['log_kernel_size'], (tuple, list))) :
        _error_popup(ParameterInputError("Either minimum_distance and 'log_kernel_size' must be correctly set\n OR 'spot_size' must be correctly set."))
    
    #Deconvolution integrity
    if values['Dense regions deconvolution'] :
        if not isinstance(values['alpha'], (float, int)) or not isinstance(values['beta'], (float, int)) :
            _error_popup(ParameterInputError("Incorrect alpha or beta parameters."))
        if type(values['gamma']) == type(None) and not isinstance(values['deconvolution_kernel'], (list, tuple)):
            _warning_popup('No gamma found; image will not be denoised before deconvolution.')
            values['gamma'] = 0
    
    #time
    if values['time stack'] :
        if type(values['time stack']) == type(None) :
            _error_popup(ParameterInputError("Incorrect time_step."))
        elif values['time stack'] == 0 :
            _error_popup(ParameterInputError("Incorrect time_step, must be > 0."))

    #channel
    if values['multichannel'] :
        ch = int(values['channel to compute'])
        if type(ch) == type(None) :
            _error_popup(ParameterInputError("Incorrect channel to compute parameter."))
        elif values['time stack'] :
            shape = values['image'].shape
            if shape[1] <= ch : _error_popup("channel to compute out of image range.")
        else :
            shape = values['image'].shape
            if shape[0] <= ch : _error_popup("channel to compute out of image range.")

    return values