"""
Contains Napari wrappers to visualise and correct spots/clusters.
"""

import napari.layers
import napari.types
import numpy as np
import napari

from napari.layers import Labels

from magicgui import widgets
from magicgui import magicgui

from bigfish.stack import check_parameter
from ._napari_widgets import cell_label_eraser, segmentation_reseter, changes_propagater, free_label_picker
from ..utils import compute_anisotropy_coef
from ..pipeline._colocalisation import spots_multicolocalisation

#Post detection

def _update_clusters(new_clusters: np.ndarray, spots: np.ndarray, voxel_size, cluster_size, shape) :
    if len(new_clusters) == 0 : return new_clusters
    if len(spots) == 0 : return np.empty(shape=(0,2+len(voxel_size)))

    if len(new_clusters[0]) in [2,3] :
        new_clusters = np.concatenate([
            new_clusters,
            np.zeros(shape=(len(new_clusters),1), dtype=int),
            np.arange(len(new_clusters), dtype=int).reshape(len(new_clusters),1)
            ],axis=1, dtype=int)

    assert len(new_clusters[0]) == 4 or len(new_clusters[0]) == 5, "Wrong number of coordinates for clusters should not happen."
    
    # Update spots clusters
    new_clusters[:,-2] = spots_multicolocalisation(new_clusters[:,:-2], spots, radius_nm= cluster_size, voxel_size=voxel_size, image_shape=shape)

    # delete too small clusters
    new_clusters = np.delete(new_clusters, new_clusters[:,-2] == 0, 0)

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
    Viewer = napari.Viewer(ndisplay=2, title= 'Spot correction', axis_labels=['z','y','x'], show= False)
    Viewer.add_image(image, scale=scale, name= "rna signal", blending= 'additive', colormap='red', contrast_limits=[image.min(), image.max()])
    other_colors = ['green', 'blue', 'gray', 'cyan', 'bop orange', 'bop purple'] * ((len(other_images)-1 // 7) + 1)
    for im, color in zip(other_images, other_colors) : 
        Viewer.add_image(im, scale=scale, blending='additive', visible=False, colormap=color, contrast_limits=[im.min(), im.max()])
    layer_offset = len(other_images)

    Viewer.add_points(  # single molecule spots; this layer can be update by user.
        spots, 
        size = 5, 
        scale=scale, 
        face_color= 'transparent', 
        opacity= 1, 
        symbol= 'disc', 
        name= 'single spots'
        )
    
    if type(clusters) != type(None) : Viewer.add_points( # cluster; this layer can be update by user.
        clusters[:,:dim], 
        size = 10, 
        scale=scale, 
        face_color= 'blue', 
        opacity= 0.7, 
        symbol= 'diamond', 
        name= 'foci', 
        features= {"spot_number" : clusters[:,dim], "id" : clusters[:,dim+1]}, 
        feature_defaults= {"spot_number" : 0, "id" : -1}
        )

    if type(cell_label) != type(None) and not np.array_equal(nucleus_label, cell_label) : Viewer.add_labels(cell_label, scale=scale, opacity= 0.2, blending= 'additive')
    if type(nucleus_label) != type(None) : Viewer.add_labels(nucleus_label, scale=scale, opacity= 0.2, blending= 'additive')
        
    Viewer.show(block=False)
    napari.run()

    new_spots = np.array(Viewer.layers['single spots'].data, dtype= int)

    if type(clusters) != type(None) :
        new_clusters = np.array(Viewer.layers['foci'].data, dtype= int)
        new_clusters = _update_clusters(new_clusters, new_spots, voxel_size=voxel_size, cluster_size=cluster_size, shape=image.shape)
    else : new_clusters = None

    return new_spots, new_clusters

# Segmentation
    


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
    Viewer = napari.Viewer(ndisplay=2, title= 'Show segmentation', axis_labels=['z','y','x'] if dim == 3 else ['y', 'x'])
    
    # Adding nuclei
    nuc_signal_layer = Viewer.add_image(nuc_image, name= "nucleus signal", blending= 'additive', colormap='blue', contrast_limits=[nuc_image.min(), nuc_image.max()])
    nuc_label_layer = Viewer.add_labels(nuc_label, opacity= 0.6, name= 'nucleus_label',)
    nuc_label_layer.preserve_labels = True
    labels_layer_list = [nuc_label_layer]
    
    #Adding cytoplasm
    if (type(cyto_label) != type(None) and not np.array_equal(cyto_label, nuc_label) ) or (type(cyto_label) != type(None) and cyto_label.max() == 0): 
        Viewer.add_image(cyto_image, name= "cytoplasm signal", blending= 'additive', colormap='red', contrast_limits=[cyto_image.min(), cyto_image.max()])
        cyto_label_layer = Viewer.add_labels(cyto_label, opacity= 0.6, name= 'cytoplasm_label')
        cyto_label_layer.preserve_labels = True
        labels_layer_list += [cyto_label_layer]

    #Adding widget
    label_eraser = cell_label_eraser(labels_layer_list)
    label_picker = free_label_picker(labels_layer_list)
    label_reseter = segmentation_reseter(labels_layer_list)
    changes_applier = changes_propagater(labels_layer_list)

    buttons_container = widgets.Container(widgets=[label_picker.widget, changes_applier.widget, label_reseter.widget], labels=False, layout='horizontal')
    tools_container = widgets.Container(
        widgets = [buttons_container, label_eraser.widget],
        labels=False,
    )
    Viewer.window.add_dock_widget(tools_container, name='SmallFish', area='left')

    #Launch Napari
    napari.run()

    new_nuc_label = Viewer.layers['nucleus_label'].data
    if 'cytoplasm_label' in Viewer.layers : 
        new_cyto_label = Viewer.layers['cytoplasm_label'].data
    else : new_cyto_label = new_nuc_label

    return new_nuc_label, new_cyto_label

def threshold_selection(
        image : np.ndarray,
        filtered_image : np.ndarray,
        threshold_slider,
        voxel_size : tuple,
        ) :
    
    """
    To view code for spot selection have a look at magicgui instance created with `detection._create_threshold_slider` which is then passed to this napari wrapper as 'threshold_slider' argument.
    """
    
    Viewer = napari.Viewer(title= "Small fish - Threshold selector", ndisplay=2, show=True)
    scale = compute_anisotropy_coef(voxel_size)
    Viewer.add_image(
        data= image,
        contrast_limits= [image.min(), image.max()],
        name= "raw signal",
        colormap= 'green',
        scale= scale,
        blending= 'additive'
    )
    Viewer.add_image(
        data= filtered_image,
        contrast_limits= [filtered_image.min(), filtered_image.max()],
        colormap= 'gray',
        scale=scale,
        blending='additive'
    )

    Viewer.window.add_dock_widget(threshold_slider, name='threshold_selector')
    threshold_slider() #First occurence with auto or entered threshold.
    
    napari.run()

    spots = Viewer.layers['single spots'].data.astype(int)
    if len(spots) == 0 :
        pass
    else :
        threshold = Viewer.layers['single spots'].properties.get('threshold')[0]

    return spots, threshold