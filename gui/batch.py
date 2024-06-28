import os
import numpy as np
import PySimpleGUI as sg
import bigfish.stack as stack
import czifile as czi

from .layout import _segmentation_layout, _detection_layout, _input_parameters_layout, _ask_channel_map_layout
from ..pipeline._preprocess import _auto_map_channels
from ..pipeline._preprocess import _check_channel_map_integrity
from ..pipeline._preprocess import MappingError


from time import sleep

def get_images(filename:str) :
    """returns filename if is image else return None"""

    supported_types = ('.tiff', '.tif', '.png', '.czi')
    if filename.endswith(supported_types) : 
        return [filename]
    else :
        return None

def get_files(path) :

    filelist = os.listdir(path)
    filelist = list(map(get_images,filelist))

    while None in filelist : filelist.remove(None)

    return filelist

def extract_files(filenames: list) :
    return sum(filenames,[])

def check_file(filename:str) :

    if filename.endswith('.czi') :
        image = czi.imread(filename)
    else :
        image = stack.read_image(filename)

    image = np.squeeze(image)

    return image.shape

def sanity_check(
        filename_list: list, 
        batch_folder : str,
        window : sg.Window, 
        progress_bar: sg.ProgressBar,
        ) :
    
    filenumber = len(filename_list)
    if filenumber == 0 :
        print("No file to check")
        progress_bar.update(current_count= 0, bar_color=('gray','gray'))
        return None
    else :
        print("{0} files to check".format(filenumber))
        progress_bar.update(current_count=0, max= filenumber)
        ref_shape = check_file(batch_folder + '/' + filename_list[0])

        print("Starting sanity check. This could take some time...")
        for i, file in enumerate(filename_list) :
            progress_bar.update(current_count= i+1, bar_color=('green','gray'))
            shape = check_file(batch_folder + '/' + file)

            if len(shape) != len(ref_shape) : #then dimension missmatch
                print("Different number of dimensions found : {0}, {1}".format(len(ref_shape), len(shape)))
                progress_bar.update(current_count=filenumber, bar_color=('red','black'))
                window= window.refresh()
                break

            window= window.refresh()

        print("Sanity check completed.")
        return None if len(shape) != len(ref_shape) else shape

def get_elmt_from_key(Tab_elmt:sg.Tab, key) -> sg.Element:
    elmt_list = sum(Tab_elmt.Rows,[])
    for elmt in elmt_list :
        if elmt.Key == key : return elmt
    raise KeyError("{0} key not found amongst {1}.".format(key, [elmt.Key for elmt in elmt_list]))

def update_master_parameters(
        Master_parameter_dict:dict,
        update_dict:dict
) :
    for parameter, is_ok in Master_parameter_dict.items() :
        elmt_to_update:sg.Element = update_dict.get(parameter)
        if type(elmt_to_update) == type(None): continue
        else :
            if is_ok : 
                text:str = elmt_to_update.DisplayText.replace('Uncorrect', 'Correct')
                color = 'green'
            else : 
                text:str = elmt_to_update.DisplayText.replace('Correct', 'Uncorrect')
                color = 'gray'

            elmt_to_update.update(value=text, text_color = color)

def update_detection_tab(
        tab_elmt:sg.Tab, 
        is_multichannel, 
        is_3D, 
        do_dense_region_deconvolution,
        do_clustering
        ) :
    
    #Acess elements
    ##Detection
    channel_to_compute = get_elmt_from_key(tab_elmt, key= 'channel to compute')
    voxel_size_z = get_elmt_from_key(tab_elmt, key= 'voxel_size_z')
    spot_size_z = get_elmt_from_key(tab_elmt, key= 'spot_size_z')
    log_kernel_size_z = get_elmt_from_key(tab_elmt, key= 'log_kernel_size_z')
    minimum_distance_z = get_elmt_from_key(tab_elmt, key= 'minimum_distance_z')

    ##Dense regions deconvolution
    alpha = get_elmt_from_key(tab_elmt, key= 'alpha')
    beta = get_elmt_from_key(tab_elmt, key= 'beta')
    gamma = get_elmt_from_key(tab_elmt, key= 'gamma')
    deconvolution_kernel_x = get_elmt_from_key(tab_elmt, key= 'deconvolution_kernel_x')
    deconvolution_kernel_y = get_elmt_from_key(tab_elmt, key= 'deconvolution_kernel_y')
    deconvolution_kernel_z = get_elmt_from_key(tab_elmt, key= 'deconvolution_kernel_z')
    cluster_size = get_elmt_from_key(tab_elmt, key= 'cluster size')
    min_number_of_spot = get_elmt_from_key(tab_elmt, key= 'min number of spots')
    nucleus_channel_signal = get_elmt_from_key(tab_elmt, key= 'nucleus channel signal')
    interactive_threshold_selector = get_elmt_from_key(tab_elmt, key= 'Interactive threshold selector')

    update_dict={
        'is_3D' : is_3D,
        'is_multichannel' : is_multichannel,
        'do_dense_region_deconvolution' : do_dense_region_deconvolution,
        'do_clustering' : do_clustering,
        'always_hidden' : False,
        'is_3D&do_denseregion' : is_3D and do_dense_region_deconvolution,
    }

    list_dict={
        'is_3D' : [voxel_size_z, spot_size_z, log_kernel_size_z, minimum_distance_z, ],
        'is_multichannel' : [channel_to_compute, nucleus_channel_signal],
        'do_dense_region_deconvolution' : [alpha,beta,gamma, deconvolution_kernel_x, deconvolution_kernel_y],
        'do_clustering' : [cluster_size, min_number_of_spot],
        'always_hidden' : [interactive_threshold_selector, ],
        'is_3D&do_denseregion' : [deconvolution_kernel_z],
        
    }

    for key, enabled in update_dict.items() :
        for elmt in list_dict.get(key) :
            elmt.update(disabled=not enabled)

def update_segmentation_tab(tab_elmt : sg.Tab, segmentation_correct_text : sg.Text, do_segmentation, is_multichannel) : 
    
    #Access elements
    cytoplasm_channel_elmt = get_elmt_from_key(tab_elmt, key= 'cytoplasm channel')
    nucleus_channel_elmt = get_elmt_from_key(tab_elmt, key= 'nucleus channel')
    
    #Update values
    tab_elmt.update(visible=do_segmentation)
    cytoplasm_channel_elmt.update(disabled = not is_multichannel)
    nucleus_channel_elmt.update(disabled = not is_multichannel)
    segmentation_correct_text.update(visible= do_segmentation)

def update_map_tab(
        tab_elmt : sg.Tab,
        is_3D,
        is_multichannel,
        last_shape,

) :
    #Acess elments
    t_element = get_elmt_from_key(tab_elmt, key= 't')
    c_element = get_elmt_from_key(tab_elmt, key= 'c')
    z_element = get_elmt_from_key(tab_elmt, key= 'z')
    automap_element = get_elmt_from_key(tab_elmt, key= 'auto-map')
    apply_element = get_elmt_from_key(tab_elmt, key= 'apply-map')

    #Update values
    t_element.update(disabled=True)
    c_element.update(disabled=not is_multichannel)
    z_element.update(disabled=not is_3D)
    automap_element.update(disabled=type(last_shape) == type(None))
    apply_element.update(disabled=type(last_shape) == type(None))

def call_auto_map(
        tab_elmt: sg.Tab,
        shape,
        is_3D,
        is_multichannel,
    ) :
    
    #Get auto map
    try :
        map = _auto_map_channels(
            is_3D_stack=is_3D,
            is_time_stack=False,
            multichannel=is_multichannel,
            image=None,
            shape=shape
            )
    
    except MappingError as e :
        sg.popup_error(e)
        return {}
    
    else :
        #Acess elemnt
        x_elmt = get_elmt_from_key(tab_elmt, 'x')
        y_elmt = get_elmt_from_key(tab_elmt, 'y')
        z_elmt = get_elmt_from_key(tab_elmt, 'z')
        c_elmt = get_elmt_from_key(tab_elmt, 'c')

        #Update values
        x_elmt.update(value=map.get('x'))
        y_elmt.update(value=map.get('y'))
        z_elmt.update(value=map.get('z'))
        c_elmt.update(value=map.get('c'))

        return map

def create_map(
        values:dict,
        is_3D:bool,
        is_multichannel:bool
        ) :
    
    maping ={
        'x': values.get('x'),
        'y': values.get('y')
    }

    if is_3D : maping['z'] = values.get('z')
    if is_multichannel : maping['c'] = values.get('c')
    
    return maping

def load(
        batch_folder:str,
) :
    if not os.path.isdir(batch_folder) :
        print("Can't open {0}".format(batch_folder))
        files_values = [[]]
        last_shape = None
        dim_number = 0
    else :
        files_values = get_files(batch_folder)
        if len(files_values) == 0 :
            last_shape = None
            dim_number = 0
        else :
            first_filename = files_values[0][0]
            last_shape = check_file(batch_folder + '/' + first_filename)
            dim_number = len(last_shape)

    
    return files_values, last_shape, dim_number


def batch_promp() :

    files_values = [[]]


    #LOAD FILES
    files_table = sg.Table(values=files_values, headings=['Filenames'], col_widths=100, max_col_width= 200, def_col_width=100, num_rows= 10, auto_size_columns=False)

    #DIMENSION SANITY
    sanity_progress = sg.ProgressBar(10, size_px=(500,10), border_width=2)
    sanity_check_button = sg.Button(
        'Check', 
        tooltip= "Will check that all files loaded have the same dimension number and that small fish is able to open them.", 
        pad=(10,0))
    sanity_header = sg.Text("Dimension sanity", font=('bold',15), pad=(0,10))
    dimension_number_text = sg.Text("Dimension number : unknown")
    

#########################################
#####   COLUMNS
#########################################

#####   Tabs

    #Input tab
    input_layout = _input_parameters_layout(
        ask_for_segmentation=True,
        is_3D_stack_preset=False,
        time_stack_preset=False,
        multichannel_preset=False,
        do_dense_regions_deconvolution_preset=False,
        do_clustering_preset=False,
        do_Napari_correction=False,
        do_segmentation_preset=False,
    )
    input_layout += [[sg.Button('Ok')]]
    input_tab = sg.Tab("Input", input_layout)

    napari_correction_elmt = get_elmt_from_key(input_tab, key='Napari correction')

    #Maptab
    map_layout = _ask_channel_map_layout(
        shape=(0,1,2,3,4),
        is_3D_stack=True,
        is_time_stack=True,
        multichannel=True,
    )
    last_shape_read = sg.Text("Last shape read : None")
    auto_map = sg.Button("auto-map", disabled=True, pad=(10,0))
    apply_map_button = sg.Button("apply", disabled=True, pad=(10,0), key='apply-map')
    map_layout += [[last_shape_read]]
    map_layout += [[auto_map, apply_map_button]]
    map_tab = sg.Tab("Map", map_layout)

    #Segmentation tab
    segmentation_layout = _segmentation_layout(multichannel=True, cytoplasm_model_preset='cyto3')
    segmentation_tab = sg.Tab("Segmentation", segmentation_layout, visible=False)

    #Detection tab
    detection_layout = _detection_layout(
        is_3D_stack=True,
        is_multichannel=True,
        do_clustering=True,
        do_dense_region_deconvolution=True,
        do_segmentation=True,
    )
    detection_tab = sg.Tab("Detection", detection_layout)

    _tab_group = sg.TabGroup([[input_tab, map_tab, segmentation_tab, detection_tab]], enable_events=True)
    tab_col = sg.Column( #Allow the tab to be scrollable
        [[_tab_group]],
        scrollable=True,
        vertical_scroll_only=True,
        s= (390,390),
        pad=((0,0),(5,5))
        )
    
#####   Launcher

    start_button =sg.Button('Start', button_color= 'green', disabled= True, pad= ((115,5),(0,10)))
    stop_button = sg.Button('Cancel', button_color= 'red', pad= ((5,115),(0,10)))
    batch_progression_bar = sg.ProgressBar(max_value=0, size_px=(340,20), bar_color= ('blue','black'), border_width=2, pad=((15,0),(30,0)))
    mapping_ok_text = sg.Text('Uncorrect mapping', text_color='gray', font='roman 14 bold', pad=((0,0),(100,5)))
    segmentation_ok_text = sg.Text('Uncorrect segmentation settings', font='roman 14 bold', pad=(0,5), text_color='gray', visible=False)
    detection_ok_text = sg.Text('Uncorrect detection settings', text_color='gray', font='roman 14 bold', pad=(0,5))
    output_ok_text = sg.Text('Uncorrect output parameters', text_color='gray', font='roman 14 bold', pad=(0,5))
    current_acquisition_text = sg.Text('0', text_color='gray', font='roman 15 bold', pad= ((150,5),(10,10)))
    total_acquisition_text = sg.Text('/ 0', text_color='gray', font='roman 15 bold', pad= ((5,150),(10,10)))

    launcher_layout = [
        [mapping_ok_text],
        [segmentation_ok_text],
        [detection_ok_text],
        [output_ok_text],
        [batch_progression_bar],
        [current_acquisition_text,total_acquisition_text],
        [start_button, stop_button],
    ]
    launch_col = sg.Column( #Allow the tab to be scrollable
        launcher_layout,
        s= (390,390),
        pad=((3,5),(5,5))
        )

    tab_dict= {
        "Input" : input_tab,
        "Segmentation" : segmentation_tab,
        "Detection" : detection_tab,
        "Map" : map_tab,
    }

#########################################
#####   Window Creation
#########################################

    layout = [
        [sg.Text("Batch Processing", font=('bold',20), pad=((300,0),(0,2)))],
        [sg.Text("Select a folder : "), sg.FolderBrowse(initial_folder=os.getcwd(), key='Batch_folder'), sg.Button('Load')],
        [files_table],
        [sanity_header, sanity_check_button, sanity_progress],
        [dimension_number_text],
        [tab_col, launch_col],
        # [sg.Output(size=(100,10), pad=(30,10))],
    ]

    window = sg.Window("small fish", layout=layout, size= (800,800), auto_size_buttons=True, auto_size_text=True)
    
    #MASTER PARAMETERS
    Master_parameters_dict ={
        '_map' : {},
        '_is_mapping_correct' : False,
        '_is_segmentation_correct' : None, # None : segmentation disabled; Then true/false if enabled
        '_is_detection_correct' : False,
        '_is_output_correct' : False,
    }
    Master_parameters_update_dict = {
        '_is_mapping_correct' : mapping_ok_text,
        '_is_segmentation_correct' : segmentation_ok_text, # None : segmentation disabled; Then true/false if enabled
        '_is_detection_correct' : detection_ok_text,
        '_is_output_correct' : output_ok_text,

    }
    loop = 0
    timeout = 1
    last_shape = None
    

#########################################
#####   Event Loop : break to close window
#########################################

    while True :
        loop +=1
        window = window.refresh()
        event, values = window.read(timeout=timeout)
        napari_correction_elmt.update(disabled=True)
        
        #Welcome message
        if loop == 1 : 
            timeout = None
            print("Welcome to small fish batch analysis. Please start by loading some files and setting parameters.")
        
        batch_folder = values.get('Batch_folder')
        is_multichanel = values.get('multichannel')
        is_3D = values.get('3D stack')
        do_segmentation = values.get('Segmentation')
        do_dense_regions_deconvolution = values.get('Dense regions deconvolution')
        do_clustering = values.get('Cluster computation')

        if type(batch_folder) != type(None)  and event == 'Load':

            files_values, last_shape, dim_number = load(batch_folder)
            files_table.update(values=files_values)
            last_shape_read.update("Last shape read : {0}".format(last_shape))
            dimension_number_text.update("Dimension number : {0}".format(dim_number))
            Master_parameters_dict['_is_mapping_correct'] = False
            update_map_tab(
                tab_elmt=tab_dict.get("Map"),
                is_3D=is_3D,
                is_multichannel=is_multichanel,
                last_shape=last_shape
            )

        elif event == 'Check' :
            filename_list = extract_files(files_values)
            last_shape = sanity_check(
                filename_list=filename_list,
                batch_folder=batch_folder,
                window=window,
                progress_bar=sanity_progress
            )
            if isinstance(last_shape,(tuple,list)) :
                dim_number = len(last_shape)
                dimension_number_text.update("Dimension number : {0}".format(dim_number))
                auto_map.update(disabled=False)
            else :
                dim_number = None
                Master_parameters_dict['_is_mapping_correct'] = False
                dimension_number_text.update("Dimension number : unknown")
                auto_map.update(disabled=True)

            last_shape_read.update("Last shape read : {0}".format(last_shape))
           
        elif event == _tab_group.key or event == 'Ok': #Tab switch in parameters
            update_segmentation_tab(
                tab_elmt=tab_dict.get("Segmentation"),
                segmentation_correct_text= segmentation_ok_text,
                do_segmentation=do_segmentation,
                is_multichannel=is_multichanel,
            )

            update_detection_tab(
                tab_elmt=tab_dict.get("Detection"),
                is_multichannel=is_multichanel,
                is_3D=is_3D,
                do_dense_region_deconvolution=do_dense_regions_deconvolution,
                do_clustering=do_clustering,
            )

            update_map_tab(
                tab_elmt=tab_dict.get("Map"),
                is_3D=is_3D,
                is_multichannel=is_multichanel,
                last_shape=last_shape
            )

        elif event == 'auto-map' :
            Master_parameters_dict['_map'] = call_auto_map(
                tab_elmt=tab_dict.get("Map"),
                shape=last_shape,
                is_3D=is_3D,
                is_multichannel=is_multichanel
            )

            Master_parameters_dict['_is_mapping_correct'] = _check_channel_map_integrity(
                maping=Master_parameters_dict['_map'],
                shape=last_shape,
                expected_dim=dim_number
                )
            
            if not Master_parameters_dict['_is_mapping_correct'] : Master_parameters_dict['_map'] = {}

        elif event == 'apply-map' :
            
            Master_parameters_dict['_map'] = create_map(
                values=values,
                is_3D=is_3D,
                is_multichannel=is_multichanel,
            )

            Master_parameters_dict['_is_mapping_correct'] = _check_channel_map_integrity(
                maping=Master_parameters_dict['_map'],
                shape=last_shape,
                expected_dim=dim_number
                )
            
            if not Master_parameters_dict['_is_mapping_correct'] : Master_parameters_dict['_map'] = {}

        elif event == 'apply-segmentation' : #TODO
            pass
        
        elif event == 'apply-detection' : #TODO
            pass
        
        elif event == 'apply-output' : #TODO
            pass

        elif event == "Cancel" :
            print(values)
        
        elif event == None :
            quit()

        #End of loop
        update_master_parameters(
            Master_parameter_dict=Master_parameters_dict,
            update_dict=Master_parameters_update_dict
        )

    window.close()