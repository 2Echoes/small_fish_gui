import napari
import numpy as np
from bigfish.stack import check_parameter
from math import ceil
from ..utils import compute_anisotropy_coef

def correct_spots(image, list_spots, voxel_size= (1,1,1)):

    check_parameter(image= np.ndarray, list_spots = list, voxel_size= (tuple,list))

    scale = compute_anisotropy_coef(voxel_size)
    try :
        Viewer = napari.Viewer(ndisplay=3, title= 'Spot correction', axis_labels=['z','y','x'], show= False)
        Viewer.add_image(image, scale=scale)

        #color prepartion
        face_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'white']
        if len(list_spots) > len(face_colors) :
            face_colors *= ceil(len(list_spots) / len(face_colors))
        
        for num, spots in enumerate(list_spots) :
            color = face_colors[num]
            check_parameter(spots = (np.ndarray, list))
            Viewer.add_points(spots, size = 5, scale=scale, face_color= color, opacity= 0.33)
        
        Viewer.show(block= True)
        corrected_spots_list = [np.array(layer.data, dtype= int) for layer in Viewer.layers[1:]]
    except Exception as error :
        corrected_spots_list = list_spots
        raise error

    finally :
        Viewer.close()
    return corrected_spots_list