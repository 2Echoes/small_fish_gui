import PySimpleGUI as sg
import os
from ..utils import check_parameter
import cellpose.models as models
from cellpose.core import use_gpu

sg.theme('DarkAmber')


def add_header(header_text) :
    """Returns [elmnt] not layout"""
    header = [sg.Text('\n{0}'.format(header_text), size= (len(header_text),3), font= 'bold 15')]
    return header


def pad_right(string, length, pad_char) :
    if len(string) >= length : return string
    else : return string + pad_char* (length - len(string))
    

def parameters_layout(parameters:'list[str]' = [], unit=None, header= None, default_values=None, size=5, opt=None) :

    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header = (str, type(None)))
    for key in parameters : check_parameter(key = str)
    max_length = len(max(parameters, key=len))

    if type(opt) == type(None) :
        opt = [False] * len(parameters)
    else :
        if len(opt) != len(parameters) : raise ValueError("Parameters and opt must be of same length.")

    if isinstance(default_values, (list, tuple)) :
        if len(default_values) != len(parameters) : raise ValueError("if default values specified it must be of equal length as parameters.")
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), 
             sg.InputText(size= size, key= parameter, default_text= value)
             
             ] for parameter,value, option in zip(parameters,default_values, opt)
        ]
    else :
        layout= [
            [sg.Text("{0}".format(pad_right(parameter, max_length, ' ')), text_color= 'green' if option else None), 
             sg.InputText(size= size, key= parameter)
             
             ] for parameter, option in zip(parameters, opt)
        ]
    
    if type(unit) == str :
        for line_id, line in enumerate(layout) :
            layout[line_id] += [sg.Text('{0}'.format(unit))]
    
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def tuple_layout(opt=None, default_dict={}, unit:dict={}, **tuples) :
    """
    tuples example : voxel_size = ['z','y','x']; will ask a tuple with 3 element default to 'z', 'y' 'x'.
    """
    if type(tuples) == type(None) : return []
    if len(tuples.keys()) == 0 : return []

    if type(opt) != type(None) :
        if not isinstance(opt, dict) : raise TypeError("opt parameter should be either None or dict type.")
        if not opt.keys() == tuples.keys() : raise ValueError("If opt is passed it is expected to have same keys as tuples dict.")
    else : 
        opt = tuples.copy()
        for key in opt.keys() :
            opt[key] = False

    for tup in tuples : 
        if not isinstance(tuples[tup], (list,tuple)) : raise TypeError()

    max_size = len(max(tuples.keys(), key=len))
    
    layout = [
        [sg.Text(pad_right(tup, max_size, ' '), text_color= 'green' if opt[option] else None)] 
        + [sg.InputText(default_text=default_dict.setdefault('{0}_{1}'.format(tup,elmnt), elmnt),key= '{0}_{1}'.format(tup, elmnt), size= 5) for elmnt in tuples[tup]]
        + [sg.Text(unit.setdefault(tup,''))] 
        for tup,option, in zip(tuples,opt)
    ]

    return layout

def path_layout(keys= [],look_for_dir = False, header=None, preset=os.getcwd()) :
    """
    If not look for dir then looks for file.
    """
    if len(keys) == 0 : return []
    check_parameter(keys= list, header = (str, type(None)))
    for key in keys : check_parameter(key = str)
    if look_for_dir : Browse = sg.FolderBrowse
    else : Browse = sg.FileBrowse

    max_length = len(max(keys, key=len))
    layout = [
        [sg.Text(pad_right(name, max_length, ' ')), Browse(key= name, initial_folder= preset)] for name in keys
        ]
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def bool_layout(parameters= [], header=None, preset=None) :
    if len(parameters) == 0 : return []
    check_parameter(parameters= list, header= (str, type(None)), preset=(type(None), list, tuple, bool))
    for key in parameters : check_parameter(key = str)
    if type(preset) == type(None) :
        preset = [False] * len(parameters)
    elif type(preset) == bool :
        preset = [preset] * len(parameters)
    else : 
        for key in preset : check_parameter(key = bool)



    max_length = len(max(parameters, key=len))
    layout = [
        [sg.Checkbox(pad_right(name, max_length, ' '), key= name, default=box_preset)] for name, box_preset in zip(parameters,preset)
    ]
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def combo_elmt(values, key, header=None, read_only=True, default_value=None) :
    """
    drop-down list
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    if type(default_value) == type(None) :
        default_value = values[0]
    elif default_value not in values :
        default_value = values[0]
    layout = [
        sg.Combo(values, default_value=default_value, readonly=read_only, key=key)
    ]
    if isinstance(header, str) :
        layout = add_header(header) + layout
    return layout

def radio_layout(values, header=None) :
    """
    Single choice buttons.
    """
    if len(values) == 0 : return []
    check_parameter(values= list, header= (str, type(None)))
    layout = [
        [sg.Radio(value, group_id= 0) for value in values]
    ]
    if isinstance(header, str) :
        layout = [add_header(header)] + layout
    return layout

def _segmentation_layout(multichannel, cytoplasm_model_preset= 'cyto2', nucleus_model_preset= 'nuclei', cytoplasm_channel_preset=0, nucleus_channel_preset=0, cyto_diameter_preset=30, nucleus_diameter_preset= 30, show_segmentation_preset= False, segment_only_nuclei_preset=False, saving_path_preset=os.getcwd(), filename_preset='cell_segmentation.png',) :
    
    USE_GPU = use_gpu()

    models_list = models.get_user_models() + models.MODEL_NAMES
    if len(models_list) == 0 : models_list = ['no model found']
    
    #Header : GPU availabality
    layout = [[sg.Text("GPU is currently "), sg.Text('ON', text_color= 'green') if USE_GPU else sg.Text('OFF', text_color= 'red')]]
    
    #cytoplasm parameters
    layout += [
        add_header("Cell Segmentation"),
        [sg.Text("Choose cellpose model for cytoplasm: \n")],
        combo_elmt(models_list, key='cyto_model_name', default_value= cytoplasm_model_preset)
                        ]
                        
    if multichannel : layout += parameters_layout(['cytoplasm channel'],default_values= [cytoplasm_channel_preset])
    layout += parameters_layout(['cytoplasm diameter'], unit= "px", default_values= [cyto_diameter_preset])
    #Nucleus parameters
    layout += [
            add_header("Nucleus segmentation"),
            [sg.Text("Choose cellpose model for nucleus: \n")],
              combo_elmt(models_list, key='nucleus_model_name', default_value= nucleus_model_preset)
                ]
    
    if multichannel : layout += parameters_layout(['nucleus channel'], default_values= [nucleus_channel_preset])
    layout += parameters_layout([ 'nucleus diameter'],unit= "px", default_values= [nucleus_diameter_preset])
    layout += bool_layout(["Segment only nuclei"], preset=segment_only_nuclei_preset)
    
    #Control plots
    layout += bool_layout(['show segmentation'], header= 'Segmentation plots', preset= show_segmentation_preset)
    layout += path_layout(['saving path'], look_for_dir=True, preset=saving_path_preset)
    layout += parameters_layout(['filename'], default_values=[filename_preset], size= 25)

    return layout

def _input_parameters_layout(
        ask_for_segmentation,
        is_3D_stack_preset,
        time_stack_preset,
        multichannel_preset,
        do_dense_regions_deconvolution_preset,
        do_clustering_preset,
        do_segmentation_preset,
        do_Napari_correction

) :
    layout_image_path = path_layout(['image path'], header= "Image")
    layout_image_path += bool_layout(['3D stack', 'multichannel'], preset= [is_3D_stack_preset, time_stack_preset, multichannel_preset])
    
    if ask_for_segmentation : 
        layout_image_path += bool_layout(
            ['Dense regions deconvolution', 'Cluster computation', 'Segmentation', 'Napari correction'], 
            preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_segmentation_preset, do_Napari_correction], 
            header= "Pipeline settings")
    else : 
        layout_image_path += bool_layout(
            ['Dense regions deconvolution', 'Cluster computation', 'Napari correction'], 
            preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_Napari_correction], 
            header= "Pipeline settings")

    return layout_image_path

def _detection_layout(
        is_3D_stack,
        is_multichannel,
        do_dense_region_deconvolution,
        do_clustering,
        do_segmentation,
        segmentation_done=False,
        default_dict={},
) :
    if is_3D_stack : dim = 3
    else : dim = 2

    #Detection
    detection_parameters = ['threshold', 'threshold penalty']
    default_detection = [default_dict.setdefault('threshold',''), default_dict.setdefault('threshold penalty', '1')]
    opt= [True, True]
    if is_multichannel : 
        detection_parameters += ['channel to compute']
        opt += [False]
        default_detection += [default_dict.setdefault('channel to compute', '')]
    
    layout = [[sg.Text("Green parameters", text_color= 'green'), sg.Text(" are optional parameters.")]]
    layout += parameters_layout(detection_parameters, header= 'Detection', opt=opt, default_values=default_detection)
    
    if dim == 2 : tuple_shape = ('y','x')
    else : tuple_shape = ('z','y','x')
    opt = {'voxel_size' : False, 'spot_size' : False, 'log_kernel_size' : True, 'minimum_distance' : True}
    unit = {'voxel_size' : 'nm', 'minimum_distance' : 'nm', 'spot_size' : 'radius(nm)', 'log_kernel_size' : 'px'}

    layout += tuple_layout(opt=opt, unit=unit, default_dict=default_dict, voxel_size= tuple_shape, spot_size= tuple_shape, log_kernel_size= tuple_shape, minimum_distance= tuple_shape)
    
    if (do_segmentation and is_multichannel) or (is_multichannel and segmentation_done):
        default_segmentation = [default_dict.setdefault('nucleus channel signal', default_dict.setdefault('nucleus channel',0))]
        layout += [[sg.Text("nucleus channel signal "), sg.InputText(default_text=default_segmentation, key= "nucleus channel signal", size= 5, tooltip= "Channel from which signal will be measured for nucleus features, \nallowing you to measure signal from a different channel than the one used for segmentation.")]]
        # layout += parameters_layout(['nucleus channel signal'], default_values=default_segmentation) + [[sg.Text("Channel from which signal will be measured for nucleus features, allowing you to measure signal from a different channel than the one used for segmentation.")]]
    
    #Deconvolution
    if do_dense_region_deconvolution :
        default_dense_regions_deconvolution = [default_dict.setdefault('alpha',0.5), default_dict.setdefault('beta',1)]
        layout += parameters_layout(['alpha', 'beta',], default_values= default_dense_regions_deconvolution, header= 'Dense regions deconvolution')
        layout += parameters_layout(['gamma'], unit= 'px', default_values= [default_dict.setdefault('gamma',5)])
        layout += tuple_layout(opt= {"deconvolution_kernel" : True}, unit= {"deconvolution_kernel" : 'px'}, default_dict=default_dict, deconvolution_kernel = tuple_shape)
    
    #Clustering
    if do_clustering :
        layout += parameters_layout(['cluster size'], unit="radius(nm)", default_values=[default_dict.setdefault('cluster size',400)])
        layout += parameters_layout(['min number of spots'], default_values=[default_dict.setdefault('min number of spots', 5)])

    

    layout += bool_layout(['Interactive threshold selector'], preset=[False])
    layout += path_layout(
        keys=['spots_extraction_folder'],
        look_for_dir=True,
        header= "Individual spot extraction",
        preset= default_dict.setdefault('spots_extraction_folder', '')
    )
    layout += parameters_layout(
        parameters=['spots_filename'],
        default_values=[default_dict.setdefault('spots_filename','spots_extraction')],
        size= 13
    )
    layout += bool_layout(
        parameters= ['do_spots_csv', 'do_spots_excel', 'do_spots_feather'],
        preset= [default_dict.setdefault('do_spots_csv',False), default_dict.setdefault('do_spots_excel',False),default_dict.setdefault('do_spots_feather',False)]
    )

    return layout

def _ask_channel_map_layout(
        shape,
        is_3D_stack,
        multichannel,
        is_time_stack,
        preset_map={},
) :
    
    x = preset_map.setdefault('x',0)
    y = preset_map.setdefault('y',0)
    z = preset_map.setdefault('z',0)
    c = preset_map.setdefault('c',0)
    t = preset_map.setdefault('t',0)

    layout = [
            add_header("Dimensions mapping"), [sg.Text("Image shape : {0}".format(shape))]
        ]
    layout += parameters_layout(['x','y'], default_values=[x,y])
    if is_3D_stack : layout += parameters_layout(['z'], default_values=[z])
    if multichannel : layout += parameters_layout(['c'], default_values=[c])
    if is_time_stack : layout += parameters_layout(['t'], default_values=[t])

    return layout