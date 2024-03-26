from cellpose.core import use_gpu
from ..gui.layout import parameters_layout, combo_layout, add_header, path_layout, bool_layout

import cellpose.models as models
import numpy as np
import bigfish.segmentation as seg
import bigfish.multistack as multistack
import bigfish.stack as stack
import PySimpleGUI as sg
import os

USE_GPU = use_gpu()

def cell_segmentation(image, cyto_model_name, nucleus_model_name, channels, cyto_diameter, nucleus_diameter) :

    nuc_channel = channels[1]
    cyto_channel = channels[0]

    if image[cyto_channel].ndim >= 3 :
        cyto = stack.maximum_projection(image[cyto_channel])
    else : 
        cyto = image[cyto_channel]
    if image[nuc_channel].ndim >= 3 :
        nuc = stack.maximum_projection(image[nuc_channel])
    else : 
        nuc = image[nuc_channel]
    
    image = np.zeros(shape=(2,) + cyto.shape)
    image[0] = cyto
    image[1] = nuc
    image = np.moveaxis(image, source=(0,1,2), destination=(2,0,1))
    print(image.shape)

    nuc_label = _segmentate_object(nuc, nucleus_model_name, nucleus_diameter, [0,0])
    cytoplasm_label = _segmentate_object(image, cyto_model_name, cyto_diameter, [1,2])
    nuc_label, cytoplasm_label = multistack.match_nuc_cell(nuc_label=nuc_label, cell_label=cytoplasm_label, single_nuc=True, cell_alone=False)

    return cytoplasm_label, nuc_label

def _segmentate_object(im, model_name, object_size_px, channels = [0,0]) :

    model = models.CellposeModel(
        gpu= USE_GPU,
        model_type= model_name,
    )

    label = model.eval(
        im,
        diameter= object_size_px,
        channels= channels,
        do_3D= False,
        )[0]
    label = np.array(label, dtype= np.int64)
    label = seg.remove_disjoint(label)
    
    return label

def _segmentation_layout(cytoplasm_channel_preset=0, nucleus_channel_preset=0, cyto_diameter_preset=30, nucleus_diameter_preset= 30, show_segmentation_preset= False, saving_path_preset=os.getcwd(), filename_preset='cell_segmentation.png') :

    models_list = models.get_user_models() + models.MODEL_NAMES
    if len(models_list) == 0 : models_list = ['no model found']
    
    #Header : GPU availabality
    layout = [[sg.Text("GPU is currently "), sg.Text('ON', text_color= 'green') if USE_GPU else sg.Text('OFF', text_color= 'red')]]
    
    #cytoplasm parameters
    layout += [add_header("Cell Segmentation", [sg.Text("Choose cellpose model for cytoplasm: \n")]),
              [combo_layout(models_list, key='cyto_model_name')]
                        ]
    layout += [parameters_layout(['cytoplasm channel', 'cytoplasm diameter'], default_values= [cytoplasm_channel_preset, cyto_diameter_preset])]
    #Nucleus parameters
    layout += [
            add_header("Nucleus segmentation",[sg.Text("Choose cellpose model for nucleus: \n")]),
              combo_layout(models_list, key='nucleus_model_name')
                ]
    layout += [parameters_layout(['nucleus channel', 'nucleus diameter'], default_values= [nucleus_channel_preset, nucleus_diameter_preset])]
    
    #Control plots
    layout += [bool_layout(['show segmentation'], header= 'Segmentation plots', preset= show_segmentation_preset)]
    layout += [path_layout(['saving path'], look_for_dir=True, preset=saving_path_preset)]
    layout += [parameters_layout(['filename'], default_values=[filename_preset], size= 25)]

    return layout

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