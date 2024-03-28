import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from ._custom_errors import MissMatchError

def reconstruct_boolean_signal(image_shape, spot_list: list):
    signal = np.zeros(image_shape, dtype= bool)
    if len(spot_list) == 0 : return signal
    dim = len(spot_list[0])

    if dim == 3 :
        Z, Y, X = list(zip(*spot_list))
        signal[Z,Y,X] = True

    else : 
        Y, X = list(zip(*spot_list))
        signal[Y,X] = True
        

    return signal

def nanometer_to_pixel(value, scale) :
    if isinstance(scale, (float,int)) : scale = [scale]
    if isinstance(value, (float,int)) : value = [value]*len(scale)
    if len(value) != len(scale) : raise ValueError("value and scale must have the same dimensionality")

    return list(np.array(value) / np.array(scale))

def _create_counting_kernel(radius_nm, voxel_size) :

    max_pixel_distance = int(max(nanometer_to_pixel(radius_nm, voxel_size)))
    kernel = np.ones(shape=[2*max_pixel_distance+1 for i in range(len(voxel_size))]) #always odd number so middle is always at [pixel_radius-1, pixel_radius-1, pixel_radius-1]
    if len(voxel_size) == 3 :
        kernel[max_pixel_distance, max_pixel_distance, max_pixel_distance] = 0
    else :
        kernel[max_pixel_distance, max_pixel_distance] = 0

    kernel = distance_transform_edt(kernel, sampling= voxel_size) <= radius_nm
    
    return kernel.astype(int)

def _spot_count_map(spots_array, radius_px, voxel_size) :
    """
    Create a map where each pixel value correspond to the number of spots closer than radius to the position.
    """

    kernel = _create_counting_kernel(radius_px, voxel_size)
    map = fftconvolve(spots_array, kernel, mode= 'same')

    return np.round(map).astype(int)

def _reconstruct_spot_signal(image_shape, spot_list: list, dim=3):
    """
    Create a map where each pixel value correspond to the number of spots located in this position.
    """
    signal = np.zeros(image_shape, dtype= int)
    unique_list, counts = np.unique(spot_list, return_counts= True, axis=0)
    if dim == 3 :
        Z, Y, X = list(zip(*unique_list))
        signal[Z,Y,X] = counts
    elif dim == 2 :
        Y, X = list(zip(*unique_list))
        signal[Y,X] = counts
    else : 
        raise ValueError("Wrong dim passed should be 2 or 3, it is {0}".format(dim))

    return signal

def spots_multicolocalisation(spots_list, anchor_list, radius_nm, image_shape, voxel_size) :

    """
    Compute the number of spots from spots_list closer than radius to a spot from anchor_list. Each spots_list spots will be counted as many times as there are anchors close enough.
    Note that the radius in nm is converted to pixel using voxel size, and rounded to the closest int value.
    
    Example in 2D
    --------

    >>> Anchors         Spots           Radius (2px)    Count
    >>> 0 0 0 0 0 0     0 X 0 0 X 0       1             0 1 0 0 0 0
    >>> 0 X 0 0 0 0     X 0 0 X 0 0     1 1 1           1 0 0 0 0 0
    >>> 0 X 0 0 0 0     X X 0 0 0 0       1             1 2 0 0 0 0     --> 5
    >>> 0 0 0 0 X 0     0 0 X 0 0 0                     0 0 0 0 0 0
    >>> 0 0 0 0 0 0     0 0 0 X 0 0                     0 0 0 0 0 0

    Parameters
    ----------
    spots_list : list
    anchor_list : list
    radius_nm : int, float
    image_shape : tuple (Z, Y, X)
    voxel_size : tuple (Z, Y, X)
    
    Returns
    -------
    Returns the list of neighbouring spot number to 'spots_list'.
    """
    if len(spots_list) == 0 or len(anchor_list) == 0 : return 0
    if len(voxel_size) != len(spots_list[0]) : raise ValueError("Dimensions missmatched; voxel_size : {0} spots : {1}".format(len(voxel_size), len(spots_list[0])))

    dim = len(voxel_size)

    anchor_array = _reconstruct_spot_signal(image_shape=image_shape, spot_list=anchor_list, dim=dim)
    count_map = _spot_count_map(anchor_array, radius_px=radius_nm, voxel_size=voxel_size)

    if dim == 3 :
        Z,Y,X = list(zip(*spots_list))
        res = list(count_map[Z,Y,X])

    if dim == 2 :
        Y,X = list(zip(*spots_list))
        res = list(count_map[Y,X])

    return res

def spots_colocalisation(image_shape, spot_list1:list, spot_list2:list, distance: float, voxel_size)-> int :
    """
    Return number of spots from spot_list1 located closer(large) than distance to at least one spot of spot_list2.

    Parameters
    ----------
        image_shape : tuple
        spot_list1 : list
        spot_list2 : list
        distance : nanometer
            distance in nanometer.
        voxel_size : (z,y,x) tuple
    """

    if len(spot_list1) == 0 or len(spot_list2) == 0 : return np.NaN
    if len(spot_list1[0]) != len(spot_list2[0]) : 
        raise MissMatchError("dimensionalities of spots 1 and spots 2 don't match.")
    
    if len(voxel_size) == 3 :
        image_shape = image_shape[-3:]
    else : 
        image_shape = image_shape[-2:]

    print("shape of signal to reconstruct : ", image_shape)

    signal2 = reconstruct_boolean_signal(image_shape, spot_list2)
    mask = np.logical_not(signal2)
    distance_map = distance_transform_edt(mask, sampling= voxel_size)

    if len(voxel_size) == 3 :
        Z,Y,X = zip(*spot_list1)
        count = (distance_map[Z,Y,X] <= distance).sum()
    else :
        Y,X = zip(*spot_list1)
        count = (distance_map[Y,X] <= distance).sum()

    return count