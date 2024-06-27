import os
import numpy as np
import PySimpleGUI as sg
import bigfish.stack as stack
import czifile as czi

from .layout import _segmentation_layout, _detection_layout


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
    else :
        print("{0} files to check".format(filenumber))
        progress_bar.update(current_count=0, max= filenumber)
        ref_shape = check_file(batch_folder + '/' + filename_list[0])

        for i, file in enumerate(filename_list) :
            print("Starting sanity check. This could take some time...")
            progress_bar.update(current_count= i+1, bar_color=('green','gray'))
            shape = check_file(batch_folder + '/' + file)

            if len(shape) != len(ref_shape) : #then dimension missmatch
                print("Different number of dimensions found : {0}, {1}".format(len(ref_shape), len(shape)))
                progress_bar.update(current_count=filenumber, bar_color=('red','black'))
                window= window.refresh()
                break

            window= window.refresh()

        print("Sanity check completed.")


def get_elmt_from_key(Tab_elmt:sg.Tab, key) :
    for elmt in sum(Tab_elmt.Rows,[]) :
        if elmt.Key == key : return elmt
    raise KeyError("{0} key not found.".format(key))

def update_detection_tab(tab_elmt:sg.Tab) : #TODO
    pass

def update_segmentation_tab(tab_elmt : sg.Tab) : #TODO
    pass

def batch_promp() :

    files_values = [[]]


    #LOAD FILES
    files_table = sg.Table(values=files_values, headings=['Filenames'], col_widths=100, max_col_width= 200, def_col_width=100, num_rows= 10, auto_size_columns=False)


    #DIMENSION SANITY
    sanity_progress = sg.ProgressBar(10, size_px=(500,10))
    sanity_check_button = sg.Button(
        'Check', 
        tooltip= "Will check that all files loaded have the same dimension number and that small fish is able to open them.", 
        pad=(10,0))
    sanity_header = sg.Text("Dimension sanity", font=('bold',15), pad=(0,10))
    dimension_number_text = sg.Text("Dimension number : unknown")

    #Segmentation tab
    segmentation_layout = _segmentation_layout(multichannel=True, cytoplasm_model_preset='cyto3')
    segmentation_tab = sg.Tab("Segmentation", segmentation_layout)

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
    _tab_group = sg.TabGroup([[segmentation_tab, detection_tab]], enable_events=True)
    tab_group = sg.Column( #Allow the tab to be scrollable
        [[_tab_group]],
        scrollable=True,
        vertical_scroll_only=True
        )
    tab_dict= {
        "Segmentation" : segmentation_tab,
        "Detection" : detection_tab
    }

    layout = [
        [sg.Text("Batch Processing", font=('bold',20), pad=((300,0),(0,2)))],
        [sg.Text("Select a folder : "), sg.FolderBrowse(initial_folder=os.getcwd(), key='Batch_folder'), sg.Button('Load')],
        [files_table],
        [sanity_header, sanity_check_button, sanity_progress],
        [dimension_number_text, sg.Checkbox('multichannel'), sg.Checkbox('3D stack')],
        [sg.Button('Ok'), sg.Button('Cancel')],
        [tab_group],
        # [sg.Output((100,30))]
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
        
        
        print("Event : ",event)

        batch_folder = values.get('Batch_folder')
        is_multichanel = values.get('multichannel')
        is_3D = values.get('3D stack')

        if type(batch_folder) != type(None)  and event == 'Load':
            if not os.path.isdir(batch_folder) :
                print("Can't open {0}".format(batch_folder))
            else :
                files_values = get_files(batch_folder)
                files_table.update(values=files_values)

        elif event == 'Check' :
            filename_list = extract_files(files_values)
            sanity_check(
                filename_list=filename_list,
                batch_folder=batch_folder,
                window=window,
                progress_bar=sanity_progress
            )
           
        elif event == 'Ok' :
            print(values)
        
        elif event == _tab_group.key : #Tab switch in parameters
            tab_name = _tab_group.get()
            print("current tab : ", tab_name)
            tab = tab_dict.get(tab_name)
            elmt = get_elmt_from_key(tab, "threshold")
            
            
            


        elif event == None :
            break