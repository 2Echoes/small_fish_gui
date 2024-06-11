"""
Contains cellpose wrappers to segmentate images.
"""

from cellpose.core import use_gpu
from skimage.measure import label
from ..gui.layout import _segmentation_layout
from ..gui import prompt, prompt_with_help, ask_cancel_segmentation

import cellpose.models as models
import numpy as np
import bigfish.multistack as multistack
import bigfish.stack as stack
import bigfish.plot as plot
import PySimpleGUI as sg
import os

def launch_segmentation(image: np.ndarray, user_parameters: dict) :
    """
    Ask user for necessary parameters and perform cell segmentation (cytoplasm + nucleus) with cellpose.

    Input
    -----
    Image : np.ndarray[c,z,y,x]
        Image to use for segmentation.

    Returns
    -------
        cytoplasm_label, nucleus_label
    """

    while True : # Loop if show_segmentation 
        #Default parameters
        cyto_model_name = user_parameters.setdefault('cyto_model_name', 'cyto2')
        cyto_size = user_parameters.setdefault('cytoplasm diameter', 180)
        cytoplasm_channel = user_parameters.setdefault('cytoplasm channel', 0)
        nucleus_model_name = user_parameters.setdefault('nucleus_model_name', 'nuclei')
        nucleus_size = user_parameters.setdefault('nucleus diameter', 130)
        nucleus_channel = user_parameters.setdefault('nucleus channel', 0)
        path = os.getcwd()
        show_segmentation = user_parameters.setdefault('show segmentation', False)
        segment_only_nuclei = user_parameters.setdefault('Segment only nuclei', False)
        filename = user_parameters['filename']
        available_channels = list(range(image.shape[0]))


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
                saving_path_preset= path,
                show_segmentation_preset=show_segmentation,
                segment_only_nuclei_preset=segment_only_nuclei,
                filename_preset=filename,
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
            path = values['saving path'] if values['saving path'] != '' else None
            show_segmentation = values['show segmentation']
            filename = values['filename'] if type(path) != type(None) else None
            channels = [cytoplasm_channel, nucleus_channel]

            relaunch= False
            #Checking integrity of parameters
            if type(cyto_model_name) != str  and not do_only_nuc:
                sg.popup('Invalid cytoplasm model name.')
                values['cyto_model_name'] = user_parameters.setdefault('cyto_model_name', 'cyto2')
                relaunch= True
            if cytoplasm_channel not in available_channels and not do_only_nuc:
                sg.popup('For given input image please select channel in {0}\ncytoplasm channel : {1}'.format(available_channels, cytoplasm_channel))
                relaunch= True
                values['cytoplasm channel'] = user_parameters.setdefault('cytoplasm channel',0)

            if type(cyto_size) not in [int, float] and not do_only_nuc:
                sg.popup("Incorrect cytoplasm size.")
                relaunch= True
                values['cytoplasm diameter'] = user_parameters.setdefault('diameter', 30)

            if type(nucleus_model_name) != str :
                sg.popup('Invalid nucleus model name.')
                values['nucleus_model_name'] = user_parameters.setdefault('nucleus_model_name', 'nuclei')
                relaunch= True
            if nucleus_channel not in available_channels :
                sg.popup('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
                relaunch= True
                values['nucleus channel'] = user_parameters.setdefault('nucleus_channel', 0)
            if type(nucleus_size) not in [int, float] :
                sg.popup("Incorrect nucleus size.")
                relaunch= True
                values['nucleus diameter'] = user_parameters.setdefault('nucleus diameter', 30)

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
                do_only_nuc=do_only_nuc
                )

        finally  : window.close()
        if show_segmentation or type(output_path) != type(None) :
            nuc_proj = image[nucleus_channel]
            im_proj = image[cytoplasm_channel]
            if im_proj.ndim == 3 :
                im_proj = stack.maximum_projection(im_proj)
            if nuc_proj.ndim == 3 :
                nuc_proj = stack.maximum_projection(nuc_proj)
            plot.plot_segmentation_boundary(nuc_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=show_segmentation, path_output=None, title= "Nucleus segmentation (blue)", remove_frame=False,)
            if type(nuc_path) != type(None) : plot.plot_segmentation_boundary(nuc_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=False, path_output=nuc_path, title= "Nucleus segmentation (blue)", remove_frame=True,)
            if not do_only_nuc : 
                plot.plot_segmentation_boundary(im_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=show_segmentation, path_output=cyto_path, title="Cytoplasm Segmentation (red)", remove_frame=False)
                if type(cyto_path) != type(None) : plot.plot_segmentation_boundary(im_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=False, path_output=cyto_path, title="Cytoplasm Segmentation (red)", remove_frame=True)
        if show_segmentation :
            layout = [
                [sg.Text("Proceed with current segmentation ?")],
                [sg.Button("Yes"), sg.Button("No")]
            ]
            
            event, values = prompt(layout=layout, add_ok_cancel=False)
            if event == "No" :
                continue

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
    return cytoplasm_label, nucleus_label, user_parameters

def cell_segmentation(image, cyto_model_name, nucleus_model_name, channels, cyto_diameter, nucleus_diameter, do_only_nuc=False) :

    nuc_channel = channels[1]
    if not do_only_nuc : 
        cyto_channel = channels[0]
        if image[cyto_channel].ndim >= 3 :
            cyto = stack.maximum_projection(image[cyto_channel])
        else : 
            cyto = image[cyto_channel]

    if image[nuc_channel].ndim >= 3 :
        nuc = stack.maximum_projection(image[nuc_channel])
    else : 
        nuc = image[nuc_channel]
    
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

def _cast_segmentation_parameters(values) :

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
