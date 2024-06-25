"""
Contains Napari wrappers to visualise and correct spots/clusters.
"""

import numpy as np
import scipy.ndimage as ndi
import napari

from napari.utils.events import Event
from napari.layers import Points
from bigfish.stack import check_parameter
from ..utils import compute_anisotropy_coef
from ._colocalisation import spots_multicolocalisation

class Points_callback :
    """
    Custom class to handle points number evolution during Napari run.
    """
    
    def __init__(self, points, next_id) -> None:
        self.points = points
        self.next_id = next_id
        self._set_callback()
    
    def __str__(self) -> str:
        string = 'Points_callback object state :\ncurrent_points_number : {0}\ncurrnet_id : {1}'.format(self.current_points_number, self.next_id)
        return string
    
    def get_points(self) :
        return self.points
    
    def get_next_id(self) : 
        return self.next_id
    
    def _set_callback(self) :
        def callback(event:Event) :

            old_points = self.get_points()
            new_points:Points = event.source.data
            features = event.source.features
            
            current_point_number = len(old_points)
            next_id = self.get_next_id()
            new_points_number = len(new_points)

            if new_points_number > current_point_number :
                features.at[new_points_number - 1, "id"] = next_id
                self.next_id += 1

            #preparing next callback
            self.points = new_points
            self._set_callback()
        self.callback = callback

def _update_clusters(new_clusters: np.ndarray, spots: np.ndarray, voxel_size, cluster_size, min_spot_number, shape) :
    if len(new_clusters) == 0 : return new_clusters
    if len(spots) == 0 : return new_clusters
    assert len(new_clusters[0]) == 4 or len(new_clusters[0]) == 5, "Wrong number of coordinates for clusters should not happen."
    
    # Update spots clusters
    if len(voxel_size) == 3 :
        new_clusters[:,-2] = spots_multicolocalisation(new_clusters[:,:3], spots, radius_nm= cluster_size, voxel_size=voxel_size, image_shape=shape)
    elif len(voxel_size) == 2 :
        new_clusters[:,-2] = spots_multicolocalisation(new_clusters[:,:2], spots, radius_nm= cluster_size, voxel_size=voxel_size, image_shape=shape)

    # delete too small clusters
        new_clusters = np.delete(new_clusters, new_clusters[:,-2] < min_spot_number, 0)

    return new_clusters

def correct_spots(image, spots, voxel_size= (1,1,1), clusters= None, cluster_size=None, min_spot_number=0, cell_label= None, nucleus_label= None, other_images =[]):
    """
    Open Napari viewer for user to visualize and corrects spots, clusters.

    Returns
    -------
        new_spots,new_clusters
    """
    check_parameter(image= np.ndarray, voxel_size= (tuple,list))
    dim = len(voxel_size)

    if dim == 3 and type(cell_label) != type(None) :
        cell_label = np.repeat(
            cell_label[np.newaxis],
            repeats= len(image),
            axis=0
        )
    if dim == 3 and type(nucleus_label) != type(None) :
        nucleus_label = np.repeat(
            nucleus_label[np.newaxis],
            repeats= len(image),
            axis=0
        )

    scale = compute_anisotropy_coef(voxel_size)
    try :
        Viewer = napari.Viewer(ndisplay=2, title= 'Spot correction', axis_labels=['z','y','x'], show= False)
        Viewer.add_image(image, scale=scale, name= "rna signal", blending= 'additive', colormap='red', contrast_limits=[image.min(), image.max()])
        other_colors = ['green', 'blue', 'gray', 'cyan', 'bop orange', 'bop purple'] * ((len(other_images)-1 // 7) + 1)
        for im, color in zip(other_images, other_colors) : 
            Viewer.add_image(im, scale=scale, blending='additive', visible=False, colormap=color, contrast_limits=[im.min(), im.max()])
        layer_offset = len(other_images)

        Viewer.add_points(spots, size = 5, scale=scale, face_color= 'green', opacity= 1, symbol= 'ring', name= 'single spots') # spots
        if type(clusters) != type(None) : Viewer.add_points(clusters[:,:dim], size = 10, scale=scale, face_color= 'blue', opacity= 0.7, symbol= 'diamond', name= 'foci', features= {"spot_number" : clusters[:,dim], "id" : clusters[:,dim+1]}, feature_defaults= {"spot_number" : 0, "id" : -1}) # cluster
        if type(cell_label) != type(None) and not np.array_equal(nucleus_label, cell_label) : Viewer.add_labels(cell_label, scale=scale, opacity= 0.2, blending= 'additive')
        if type(nucleus_label) != type(None) : Viewer.add_labels(nucleus_label, scale=scale, opacity= 0.2, blending= 'additive')
        
        #prepare cluster update
        if type(clusters) != type(None) : 
            next_cluster_id = clusters[-1,-1] + 1 if len(clusters) > 0 else 1
            _callback = Points_callback(points=clusters[:dim], next_id=next_cluster_id)
            points_callback = Viewer.layers[2 + layer_offset].events.data.connect((_callback, 'callback'))
        Viewer.show(block=False)
        napari.run()
        

        new_spots = np.array(Viewer.layers[1 + layer_offset].data, dtype= int)

        if type(clusters) != type(None) :
            if len(clusters) > 0 : 
                new_clusters = np.concatenate([
                    np.array(Viewer.layers[2 + layer_offset].data, dtype= int),
                    np.array(Viewer.layers[2 + layer_offset].features, dtype= int)
                ],
                axis= 1)

                new_clusters = _update_clusters(new_clusters, new_spots, voxel_size=voxel_size, cluster_size=cluster_size, min_spot_number=min_spot_number, shape=image.shape)
        else : new_clusters = None

    except Exception as error :
        new_spots = spots
        new_clusters = clusters
        raise error

    return new_spots, new_clusters

def show_segmentation(
        nuc_image : np.ndarray,
        nuc_label : np.ndarray,
        cyto_image : np.ndarray = None,
        cyto_label : np.ndarray = None,
) :
    dim = nuc_image.ndim
    
    if type(cyto_image) != type(None) :
        if cyto_image.ndim != nuc_image.ndim : raise ValueError("Cyto and Nuc dimensions missmatch.")
        if type(cyto_label) == type(None) : raise ValueError("If cyto image is passed cyto label must be passed too.")

    if dim == 3 and nuc_label.ndim == 2 :
        nuc_label = np.repeat(
            nuc_label[np.newaxis],
            repeats= len(nuc_image),
            axis=0
        )
    if type(cyto_label) != type(None) :
        
        if type(cyto_image) == type(None) : raise ValueError("If cyto label is passed cyto image must be passed too.")
        
        if dim == 3 and cyto_label.ndim == 2 :
            cyto_label = np.repeat(
                cyto_label[np.newaxis],
                repeats= len(nuc_image),
                axis=0
            )

    #Init Napari viewer
    Viewer = napari.Viewer(ndisplay=2, title= 'Show segmentation', axis_labels=['z','y','x'] if dim == 3 else ['y', 'x'], show= False)
    
    # Adding channels
    Viewer.add_image(nuc_image, name= "nucleus signal", blending= 'additive', colormap='blue', contrast_limits=[nuc_image.min(), nuc_image.max()])
    Viewer.add_labels(nuc_label, opacity= 0.5, blending= 'additive')
    
    #Adding labels
    if type(cyto_image) != type(None) : Viewer.add_image(cyto_image, name= "cytoplasm signal", blending= 'additive', colormap='red', contrast_limits=[cyto_image.min(), cyto_image.max()])
    if type(cyto_label) != type(None) : Viewer.add_labels(cyto_label, opacity= 0.4, blending= 'additive')
    
    #Launch Napari
    Viewer.show(block=False)
    napari.run()

    new_nuc_label = Viewer.layers[1].data
    if type(cyto_label) != type(None) : new_cyto_label = Viewer.layers[3].data

    return new_nuc_label, new_cyto_label



def threshold_selection(
        image : np.ndarray,
        filtered_image : np.ndarray,
        threshold_slider,
        voxel_size : tuple,
        ) :
    
    """
    To view code for spot selection please have a look at magicgui instance created with `detection._create_threshold_slider` which is then passed to this napari wrapper as 'threshold_slider' argument.
    """
    
    
    Viewer = napari.Viewer(title= "Small fish - Threshold selector", ndisplay=2, show=True)
    Viewer.add_image(
        data= image,
        contrast_limits= [image.min(), image.max()],
        name= "raw signal",
        colormap= 'green',
        scale= voxel_size,
        blending= 'additive'
    )
    Viewer.add_image(
        data= filtered_image,
        contrast_limits= [filtered_image.min(), filtered_image.max()],
        colormap= 'gray',
        scale=voxel_size,
        blending='additive'
    )

    Viewer.window.add_dock_widget(threshold_slider, name='threshold_selector')
    threshold_slider() #First occurence with auto or entered threshold.
    napari.run()

    spots = Viewer.layers[-1].data.astype(int)
    if len(spots) == 0 :
        threshold = filtered_image.max()
    else :
        threshold = Viewer.layers[-1].properties.get('threshold')[0]

    return spots, threshold