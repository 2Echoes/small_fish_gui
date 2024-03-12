from cellpose.core import use_gpu
from ..gui.layout import parameters_layout, combo_layout, add_header, path_layout


import cellpose.models as models
import numpy as np
import bigfish.segmentation as seg
import bigfish.plot as plot
import bigfish.multistack as multistack
import PySimpleGUI as sg


USE_GPU = use_gpu()

def cell_segmentation(image, model_name, channels, obj_diameter, output_path=None) :

    nuc_channel = channels[1]
    nuc_label = seg._segmentate_object(image[nuc_channel], model_name, obj_diameter, [0,0])
    cytoplasm_label = seg._segmentate_object(image, model_name, obj_diameter, channels)
    cytoplasm_label, nuc_label = multistack.match_nuc_cell(nuc_label=nuc_channel, cell_label=cytoplasm_label, single_nuc=True, cell_alone=False)

    return cytoplasm_label, nuc_label

def _segmentate_object(im, model_name, object_size_px, channels = [0,0]) :

    model = models.CellposeModel(
        gpu= USE_GPU,
        pretrained_model= model_name,
    )

    label = model.eval(
        im,
        diameter= object_size_px,
        channels= channels,
        progress= True
        )[0]
    label = np.array(label, dtype= np.int64)
    label = seg.remove_disjoint(label)
    
    return label

def _segmentation_layout() :

    models_list = models.get_user_models()
    layout = [add_header("Cell Segmentation", [sg.Text("Choose cellpose model : \n")]),
              [combo_layout(models_list, key='model_name')]
                        ]
    layout += parameters_layout(['cytoplasm channel', 'nucleus channel', 'object diameter'], default_values= [0, 0, 30])
    layout += path_layout(['saving path'], look_for_dir=True)

    return layout
