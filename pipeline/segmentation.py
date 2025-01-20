"""
Contains cellpose wrappers to segmentate images.
"""

from cellpose.core import use_gpu
from skimage.measure import label
from ..hints import pipeline_parameters
from ..gui.layout import _segmentation_layout
from ..gui import prompt, prompt_with_help, ask_cancel_segmentation
from ..interface import open_image
from ..gui.napari_visualiser import show_segmentation as napari_show_segmentation
from .utils import from_label_get_centeroidscoords
from ._preprocess import ask_input_parameters
from ._preprocess import map_channels, reorder_shape, reorder_image_stack
from matplotlib.colors import ListedColormap

import matplotlib as mpl
import cellpose.models as models
import numpy as np
import bigfish.multistack as multistack
import bigfish.stack as stack
import bigfish.plot as plot
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import os

def launch_segmentation(user_parameters: pipeline_parameters, nucleus_label, cytoplasm_label) :
    """
    Ask user for necessary parameters and perform cell segmentation (cytoplasm + nucleus) with cellpose.

    Input
    -----
    Image : np.ndarray[c,z,y,x]
        Image to use for segmentation.

    Returns
    -------
        cytoplasm_label, nucleus_label, user_parameters
    """

    segmentation_parameters = user_parameters.copy()

    #Ask for image parameters
    new_parameters = ask_input_parameters(ask_for_segmentation= True) #The image is open and stored inside user_parameters
    if type(new_parameters) == type(None) : #if user clicks 'Cancel'
        return nucleus_label , cytoplasm_label, user_parameters
    else :
        segmentation_parameters.update(new_parameters)

    map_ = map_channels(segmentation_parameters)
    if type(map_) == type(None) : #User clicks Cancel 
        return nucleus_label, cytoplasm_label, user_parameters
    segmentation_parameters['reordered_shape'] = reorder_shape(segmentation_parameters['shape'], map_)
    image = reorder_image_stack(map_, segmentation_parameters['image'])

    while True : # Loop if show_segmentation 
        #Default parameters
        cyto_model_name = segmentation_parameters.setdefault('cyto_model_name', 'cyto3')
        cyto_size = segmentation_parameters.setdefault('cytoplasm diameter', 180)
        cytoplasm_channel = segmentation_parameters.setdefault('cytoplasm channel', 0)
        nucleus_model_name = segmentation_parameters.setdefault('nucleus_model_name', 'nuclei')
        nucleus_size = segmentation_parameters.setdefault('nucleus diameter', 130)
        nucleus_channel = segmentation_parameters.setdefault('nucleus channel', 0)
        other_nucleus_image = segmentation_parameters.setdefault('other_nucleus_image',None)
        path = os.getcwd()
        show_segmentation = segmentation_parameters.setdefault('show segmentation', False)
        segment_only_nuclei = segmentation_parameters.setdefault('Segment only nuclei', False)
        filename = segmentation_parameters['filename']
        available_channels = list(range(image.shape[0]))
        multichannel = segmentation_parameters.get('is_multichannel')


    #Ask user for parameters
    #if incorrect parameters --> set relaunch to True
        while True :
            layout = _segmentation_layout(
                cytoplasm_model_preset = cyto_model_name,
                cytoplasm_channel_preset= cytoplasm_channel,
                nucleus_model_preset = nucleus_model_name,
                nucleus_channel_preset= nucleus_channel,
                cyto_diameter_preset= cyto_size,
                nucleus_diameter_preset= nucleus_size,
                other_nucleus_image_preset=other_nucleus_image,
                saving_path_preset= path,
                show_segmentation_preset=show_segmentation,
                segment_only_nuclei_preset=segment_only_nuclei,
                filename_preset=filename,
                multichannel=multichannel,
            )

            event, values = prompt_with_help(layout, help='segmentation')
            if event == 'Cancel' :
                cancel_segmentation = ask_cancel_segmentation()

                if cancel_segmentation :
                    return None, None, user_parameters
                else : 
                    continue

            #Extract parameters
            values = _cast_segmentation_parameters(values)
            do_only_nuc = values['Segment only nuclei']
            cyto_model_name = values['cyto_model_name']
            cyto_size = values['cytoplasm diameter']
            cytoplasm_channel = values['cytoplasm channel']
            nucleus_model_name = values['nucleus_model_name']
            nucleus_size = values['nucleus diameter']
            nucleus_channel = values['nucleus channel']
            other_nucleus_image = values['other_nucleus_image']
            path = values['saving path'] if values['saving path'] != '' else None
            show_segmentation = values['show segmentation']
            filename = values['filename'] if type(path) != type(None) else None
            channels = [cytoplasm_channel, nucleus_channel] if multichannel else [...,...]

            relaunch= False
            #Checking integrity of parameters
            if type(cyto_model_name) != str  and not do_only_nuc:
                sg.popup('Invalid cytoplasm model name.')
                values['cyto_model_name'] = user_parameters.setdefault('cyto_model_name', 'cyto2')
                relaunch= True
            if multichannel :
                if cytoplasm_channel not in available_channels and not do_only_nuc:
                    sg.popup('For given input image please select channel in {0}\ncytoplasm channel : {1}'.format(available_channels, cytoplasm_channel))
                    relaunch= True
                    values['cytoplasm channel'] = user_parameters.setdefault('cytoplasm channel',0)
            else :
                cytoplasm_channel = ...

            if type(cyto_size) not in [int, float] and not do_only_nuc:
                sg.popup("Incorrect cytoplasm size.")
                relaunch= True
                values['cytoplasm diameter'] = user_parameters.setdefault('diameter', 30)

            if type(nucleus_model_name) != str :
                sg.popup('Invalid nucleus model name.')
                values['nucleus_model_name'] = user_parameters.setdefault('nucleus_model_name', 'nuclei')
                relaunch= True
            
            if multichannel :
                if nucleus_channel not in available_channels :
                    sg.popup('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
                    relaunch= True
                    values['nucleus channel'] = user_parameters.setdefault('nucleus_channel', 0)
            else : 
                nucleus_channel = ...

            if type(nucleus_size) not in [int, float] :
                sg.popup("Incorrect nucleus size.")
                relaunch= True
                values['nucleus diameter'] = user_parameters.setdefault('nucleus diameter', 30)
            if other_nucleus_image != '' :
                if not os.path.isfile(other_nucleus_image) :
                    sg.popup("Nucleus image is not a file.")
                    relaunch=True
                    values['other_nucleus_image'] = None
                else :
                    try :
                        nucleus_image = open_image(other_nucleus_image)
                    except Exception as e :
                        sg.popup("Could not open image.\n{0}".format(e))
                        relaunch=True
                        values['other_nucleus_image'] = user_parameters.setdefault('other_nucleus_image', None)
                    else :
                        if nucleus_image.ndim != image.ndim - multichannel :
                            sg.popup("Nucleus image dimension missmatched. Expected same dimension as cytoplasm_image for monochannel or same dimension as cytoplasm_image -1 for multichannel\ncytoplasm dimension : {0}, nucleus dimension : {1}".format(image.ndim, nucleus_image.ndim))
                            nucleus_image = None
                            relaunch=True
                            values['other_nucleus_image'] = user_parameters.setdefault('other_nucleus_image', None)
                        
                        elif nucleus_image.shape != image[cytoplasm_channel].shape :
                            sg.popup("Nucleus image shape missmatched. Expected same shape as cytoplasm_image \ncytoplasm shape : {0}, nucleus shape : {1}".format(image[cytoplasm_channel].shape, nucleus_image.shape))
                            nucleus_image = None
                            relaunch=True
                            values['other_nucleus_image'] = user_parameters.setdefault('other_nucleus_image', None)

            else :
                nucleus_image = None

            user_parameters.update(values)
            if not relaunch : break

        #Launching segmentation
        waiting_layout = [
            [sg.Text("Running segmentation...")]
        ]
        window = sg.Window(
            title= 'small_fish',
            layout= waiting_layout,
            grab_anywhere= True,
            no_titlebar= False
        )

        window.read(timeout= 30, close= False)

        try :
            if type(path) != type(None) and filename != '':
                output_path = path + '/' + filename
                nuc_path = output_path + "_nucleus_segmentation"
                cyto_path = output_path + "_cytoplasm_segmentation"
            else :
                output_path = None
                nuc_path = None
                cyto_path = None


            cytoplasm_label, nucleus_label = cell_segmentation(
                image,
                cyto_model_name= cyto_model_name,
                cyto_diameter= cyto_size,
                nucleus_model_name= nucleus_model_name,
                nucleus_diameter= nucleus_size,
                channels=channels,
                do_only_nuc=do_only_nuc,
                external_nucleus_image = nucleus_image,
                )

        finally  : window.close()

        if show_segmentation :
            nucleus_label, cytoplasm_label = napari_show_segmentation(
                nuc_image=image[nucleus_channel] if type(nucleus_image) == type(None) else nucleus_image,
                nuc_label= nucleus_label,
                cyto_image=image[cytoplasm_channel],
                cyto_label=cytoplasm_label,
            )

            if nucleus_label.ndim == 3 : nucleus_label = np.max(nucleus_label, axis=0)
            if cytoplasm_label.ndim == 3 : cytoplasm_label = np.max(cytoplasm_label, axis=0)

            layout = [
                [sg.Text("Proceed with current segmentation ?")],
                [sg.Button("Yes"), sg.Button("No")]
            ]
            
            event, values = prompt(layout=layout, add_ok_cancel=False, add_scrollbar=False)
            if event == "No" :
                continue

        if type(output_path) != type(None) :
            
            #Get backgrounds
            nuc_proj = image[nucleus_channel]
            im_proj = image[cytoplasm_channel]
            if im_proj.ndim == 3 :
                im_proj = stack.maximum_projection(im_proj)
            if nuc_proj.ndim == 3 :
                nuc_proj = stack.maximum_projection(nuc_proj)
            
            #Call plots
            plot.plot_segmentation_boundary(nuc_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=False, path_output=nuc_path, title= "Nucleus segmentation (blue)", remove_frame=True,)
            if not do_only_nuc : 
                plot.plot_segmentation_boundary(im_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=False, path_output=cyto_path, title="Cytoplasm Segmentation (red)", remove_frame=True)
            plot_labels(
                nucleus_label,
                path_output=output_path + "_nucleus_label_map.png",
                show=False
                )
            if not do_only_nuc : 
                plot_labels(
                    cytoplasm_label,
                    path_output=output_path + "_cytoplasm_label_map.png",
                    show=False
                    )



        if cytoplasm_label.max() == 0 : #No cell segmented
            layout = [
            [sg.Text("No cell segmented. Proceed anyway ?")],
            [sg.Button("Yes"), sg.Button("No")]
        ]
            event, values = prompt(layout=layout, add_ok_cancel=False)
            if event == "Yes" :
                return None, None, user_parameters
        else :
            break


        
    user_parameters.update(values)
    return nucleus_label, cytoplasm_label, user_parameters

def cell_segmentation(
        image, cyto_model_name, 
        nucleus_model_name, 
        channels, cyto_diameter, 
        nucleus_diameter, 
        do_only_nuc=False,
        external_nucleus_image = None,
        ) :

    nuc_channel = channels[1]
    if not do_only_nuc : 
        cyto_channel = channels[0]
        if image[cyto_channel].ndim >= 3 :
            cyto = stack.maximum_projection(image[cyto_channel])
        else : 
            cyto = image[cyto_channel]

    if type(external_nucleus_image) != type(None) :
        nuc = external_nucleus_image
    else :
        nuc = image[nuc_channel]

    if nuc.ndim >= 3 :
        nuc = stack.maximum_projection(nuc)
    
    if not do_only_nuc :
        image = np.zeros(shape=(2,) + cyto.shape)
        image[0] = cyto
        image[1] = nuc
        image = np.moveaxis(image, source=(0,1,2), destination=(2,0,1))

    nuc_label = _segmentate_object(nuc, nucleus_model_name, nucleus_diameter, [0,0])
    if not do_only_nuc :
        cytoplasm_label = _segmentate_object(image, cyto_model_name, cyto_diameter, [1,2])
        nuc_label, cytoplasm_label = multistack.match_nuc_cell(nuc_label=nuc_label, cell_label=cytoplasm_label, single_nuc=True, cell_alone=False)
    else :
        cytoplasm_label = nuc_label

    return cytoplasm_label, nuc_label

def _segmentate_object(im, model_name, object_size_px, channels = [0,0]) :

    model = models.CellposeModel(
        gpu= use_gpu(),
        model_type= model_name,
    )

    label = model.eval(
        im,
        diameter= object_size_px,
        channels= channels,
        do_3D= False,
        )[0]
    label = np.array(label, dtype= np.int64)
    label = remove_disjoint(label)
    
    return label

def _cast_segmentation_parameters(values:dict) :

    values.setdefault('cytoplasm channel',0)
    values.setdefault('nucleus channel',0)

    if values['cyto_model_name'] == '' :
        values['cyto_model_name'] = None

    if values['nucleus_model_name'] == '' :
        values['nucleus_model_name'] = None

    try : #cytoplasm channel
        values['cytoplasm channel'] = int(values['cytoplasm channel'])
    except ValueError :
        pass

    try : #nucleus channel
        values['nucleus channel'] = int(values['nucleus channel'])
    except ValueError :
        pass

    try : #object size
        values['cytoplasm diameter'] = float(values['cytoplasm diameter'])
    except ValueError :
        pass

    try : #object size
        values['nucleus diameter'] = float(values['nucleus diameter'])
    except ValueError :
        pass
    
    return values

def remove_disjoint(image):
    """
    *CODE FROM BIG-FISH (LICENCE IN __INIT__.PY) IMPORTED TO AVOID IMPORT ERROR WHEN BIGFISH SEGMENTATION MODULE INITIALISES : try to import deprecated module for python 3.8

    For each instances with disconnected parts, keep the larger one.

    Parameters
    ----------
    image : np.ndarray, np.int, np.uint or bool
        Labelled image with shape (z, y, x) or (y, x).

    Returns
    -------
    image_cleaned : np.ndarray, np.int or np.uint
        Cleaned image with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.int32, np.int64, bool])

    # handle boolean array
    cast_to_bool = False
    if image.dtype == bool:
        cast_to_bool = bool
        image = image.astype(np.uint8)

    # initialize cleaned labels
    image_cleaned = np.zeros_like(image)

    # loop over instances
    max_label = image.max()
    for i in range(1, max_label + 1):

        # get instance mask
        mask = image == i

        # check if an instance is labelled with this value
        if mask.sum() == 0:
            continue

        # get an index for each disconnected part of the instance
        labelled_mask = label(mask)
        indices = sorted(list(set(labelled_mask.ravel())))
        if 0 in indices:
            indices = indices[1:]

        # keep the largest part of the instance
        max_area = 0
        mask_instance = None
        for j in indices:
            mask_part_j = labelled_mask == j
            area_j = mask_part_j.sum()
            if area_j > max_area:
                max_area = area_j
                mask_instance = mask_part_j

        # add instance in the final label
        image_cleaned[mask_instance] = i

    if cast_to_bool:
        image_cleaned = image_cleaned.astype(bool)

    return image_cleaned

def plot_segmentation(
        cyto_image : np.ndarray, 
        cyto_label : np.ndarray, 
        nuc_image : np.ndarray, 
        nuc_label : np.ndarray,
        path :str, 
        do_only_nuc=False
        ) :

    if nuc_image.ndim == 3 :
        nuc_image = np.max(nuc_image,axis=0)
    
    plot.plot_segmentation_boundary(
        image=nuc_image,
        nuc_label= nuc_label,
        boundary_size= 3,
        contrast=True,
        path_output=path + "_nuclei_segmentation.png",
        show=False,
    )


    if not do_only_nuc :
        if cyto_image.ndim == 3 :
            cyto_image = np.max(cyto_image,axis=0)
    
        plot.plot_segmentation_boundary(
            image=cyto_image,
            cell_label= cyto_label,
            nuc_label= nuc_label,
            boundary_size= 3,
            contrast=True,
            path_output=path + "_cytoplasm_segmentation.png",
            show=False,
        )

def plot_labels(labelled_image: np.ndarray, path_output:str = None, show= True, axis= False, close= True):
    """
    Plot a labelled image and indicate the label number at the center of each region.
    """
    stack.check_parameter(labelled_image = (np.ndarray, list), show = (bool))
    if isinstance(labelled_image, np.ndarray) : 
        stack.check_array(labelled_image, ndim= 2)
        labelled_image = [labelled_image]
    
    #Setting a colormap with background to white so all cells can be visible
    viridis = mpl.colormaps['viridis'].resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[0, :] = white
    newcmp = ListedColormap(newcolors)

    plt.figure(figsize= (10,10))
    rescaled_image = stack.rescale(np.array(labelled_image[0], dtype= np.int32), channel_to_stretch= 0)
    rescaled_image[rescaled_image == 0] = -100
    plot = plt.imshow(rescaled_image, cmap=newcmp)
    plot.axes.get_xaxis().set_visible(axis)
    plot.axes.get_yaxis().set_visible(axis)
    plt.tight_layout()

    for index in range(0, len(labelled_image)) :
        centroid_dict = from_label_get_centeroidscoords(labelled_image[index])
        labels = centroid_dict["label"]
        Y = centroid_dict["centroid-0"]
        X = centroid_dict["centroid-1"]
        centroids = zip(Y,X)

        for label in labels :
            y,x = next(centroids)
            y,x = round(y), round(x)
            an = plt.annotate(str(label), [round(x), round(y)])

    if not axis : plt.cla
    if show : plt.show()
    if path_output != None :
        stack.check_parameter(path_output = (str))
        plt.savefig(path_output)
    if close : plt.close()

    return plot