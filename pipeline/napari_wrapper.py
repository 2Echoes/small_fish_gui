import napari
from ..utils import compute_anisotropy_coef

def correct_spots(image, spots, voxel_size= (1,1,1)):
    scale = compute_anisotropy_coef(voxel_size)
    try :
        Viewer = napari.Viewer(title= 'Spot correction', axis_labels=['z','y','x'], show= False)
        Viewer.add_image(image, scale=scale)
        corrected_spots = Viewer.add_points(spots, size = 2, scale=scale)
        Viewer.show(block= True)
    finally :
        Viewer.close()
    return corrected_spots