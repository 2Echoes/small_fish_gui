import os
import numpy as np
import PySimpleGUI as sg
import bigfish.stack as stack
import czifile as czi

from .layout import _segmentation_layout, _detection_layout, _input_parameters_layout, _ask_channel_map_layout


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
        'always_hidden' : False
    }

    list_dict={
        'is_3D' : [voxel_size_z, spot_size_z, log_kernel_size_z, minimum_distance_z, deconvolution_kernel_z],
        'is_multichannel' : [channel_to_compute, nucleus_channel_signal],
        'do_dense_region_deconvolution' : [alpha,beta,gamma],
        'do_clustering' : [cluster_size, min_number_of_spot],
        'always_hidden' : [interactive_threshold_selector]
        
    }

    for key, enabled in update_dict.items() :
        for elmt in list_dict.get(key) :
            elmt.update(disabled=not enabled)
    


def update_segmentation_tab(tab_elmt : sg.Tab, do_segmentation, is_multichannel) : #TODO
    
    #Access elements
    cytoplasm_channel_elmt = get_elmt_from_key(tab_elmt, key= 'cytoplasm channel')
    nucleus_channel_elmt = get_elmt_from_key(tab_elmt, key= 'nucleus channel')
    
    #Update values
    tab_elmt.update(visible=do_segmentation)
    cytoplasm_channel_elmt.update(disabled = not is_multichannel)
    nucleus_channel_elmt.update(disabled = not is_multichannel)

def update_map_tab() :
    #TODO
    pass

def batch_promp() :

    files_values = [[]]


    #LOAD FILES
    files_table = sg.Table(values=files_values, headings=['Filenames'], col_widths=100, max_col_width= 200, def_col_width=100, num_rows= 10, auto_size_columns=False)

    #Start&Stop
    start_button =sg.Button('Start', button_color= 'green', disabled= True)
    stop_button = sg.Button('Cancel', button_color= 'red')

    #DIMENSION SANITY
    sanity_progress = sg.ProgressBar(10, size_px=(500,10))
    sanity_check_button = sg.Button(
        'Check', 
        tooltip= "Will check that all files loaded have the same dimension number and that small fish is able to open them.", 
        pad=(10,0))
    sanity_header = sg.Text("Dimension sanity", font=('bold',15), pad=(0,10))
    dimension_number_text = sg.Text("Dimension number : unknown")

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

    #Maptab
    map_layout = _ask_channel_map_layout(
        shape=(0,1,2,3,4),
        is_3D_stack=True,
        is_time_stack=True,
        multichannel=True,
    )
    last_shape_read = sg.Text("Last shape read : None")
    auto_map = sg.Button("auto-map", disabled=True, pad=(10,0))
    map_layout += [[last_shape_read, auto_map]]
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
    
        

    #TABS
    _tab_group = sg.TabGroup([[input_tab, map_tab, segmentation_tab, detection_tab]], enable_events=True)
    tab_group = sg.Column( #Allow the tab to be scrollable
        [[_tab_group]],
        scrollable=True,
        vertical_scroll_only=True,
        pad=(150,5)
        )
    tab_dict= {
        "Input" : input_tab,
        "Segmentation" : segmentation_tab,
        "Detection" : detection_tab,
        "Map" : map_tab,
    }

    layout = [
        [sg.Text("Batch Processing", font=('bold',20), pad=((300,0),(0,2)))],
        [sg.Text("Select a folder : "), sg.FolderBrowse(initial_folder=os.getcwd(), key='Batch_folder'), sg.Button('Load')],
        [files_table],
        [sanity_header, sanity_check_button, sanity_progress],
        [dimension_number_text],
        [tab_group],
        [sg.Output(size=(100,10), pad=(30,10))],
        [start_button, stop_button],
    ]

    window = sg.Window("small fish", layout=layout, size= (800,800), auto_size_buttons=True, auto_size_text=True)
    loop = 0
    timeout = 1
    while True :
        loop +=1
        window = window.refresh()
        event, values = window.read(timeout=timeout)
        
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
            if not os.path.isdir(batch_folder) :
                print("Can't open {0}".format(batch_folder))
            else :
                files_values = get_files(batch_folder)
                files_table.update(values=files_values)

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
                dimension_number_text.update("Dimension number : unknown")
                auto_map.update(disabled=True)

            last_shape_read.update("Last shape read : {0}".format(last_shape))
           
        
        elif event == _tab_group.key or event == 'Ok': #Tab switch in parameters
            update_segmentation_tab(
                tab_elmt=tab_dict.get("Segmentation"),
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

        elif event == 'auto-map' :
            #TODO
            pass

        # elif event == 'apply' (map) #TODO

        # elif event == 'check parameters' -> un/lock start #TODO
            
        elif event == "Cancel" :
            print(values)
        elif event == None :
            quit()