import PySimpleGUI as sg
import pandas as pd
import os
import numpy as np
from typing import Literal, Union, Any
from .layout import path_layout, parameters_layout, bool_layout, tuple_layout, combo_elmt, add_header, path_layout, radio_layout
from ..interface import open_image, check_format, FormatError
from .help_module import ask_help

def prompt(layout, add_ok_cancel=True, timeout=None, timeout_key='TIMEOUT_KEY', add_scrollbar=True) :
    """
    Default event : 'Ok', 'Cancel'
    """
    if add_ok_cancel : layout += [[sg.Button('Ok'), sg.Button('Cancel')]]

    if add_scrollbar :
        size = (400,500)
        col_elmt = sg.Column(layout, scrollable=True, vertical_scroll_only=True, size=size)
        layout = [[col_elmt]]
    else :
        size = (None,None)
    
    window = sg.Window('small fish', layout=layout, margins=(10,10), size=size, resizable=True)
    event, values = window.read(timeout=timeout, timeout_key=timeout_key)
    if event == None : 
        window.close()
        quit()

    elif event == 'Cancel' :
        window.close()
        return event,{}
    else : 
        window.close()
        return event, values

def prompt_with_help(layout, help =None, add_scrollbar=True, vertical_scroll_only=True) :
    layout += [[]]
    layout += [[sg.Button('Ok'), sg.Button('Cancel')]]
    
    if add_scrollbar :
        size = (400,500)
        col_elmt = sg.Column(layout, scrollable=True, vertical_scroll_only=vertical_scroll_only, size=size)
        layout = [[col_elmt]]
    else :
        size = (None,None)

    window = sg.Window('small fish', layout=layout, size=size, resizable=True)
    while True :
        event, values = window.read()
        if event == None :
            window.close()
            quit()

        elif event == 'Ok': 
            window.close()
            return event, values
        elif event == 'Help' :
            ask_help(chapter= help)
        
        else:
            window.close()
            return event,{}

def input_image_prompt(
        is_3D_stack_preset=False,
        multichannel_preset = False,
        do_dense_regions_deconvolution_preset= False,
        do_clustering_preset = False,
        do_Napari_correction= False,
) :
    """
        Keys :
        - 'image path'
        - 'is_3D_stack'
        - 'time stack'
        - 'is_multichannel'
        - 'do_dense_regions_deconvolution'
        - 'Segmentation'
        - 'show_napari_corrector'

    Returns Values

    """
    layout_image_path = path_layout(['image path'], header= "Image")
    layout_image_path += bool_layout(['3D stack', 'Multichannel stack'],keys= ['is_3D_stack', 'is_multichannel'], preset= [is_3D_stack_preset, multichannel_preset])
    
    if type(do_dense_regions_deconvolution_preset) != type(None) and type(do_clustering_preset) != type(None) and type(do_Napari_correction) != type(None): 
        layout_image_path += bool_layout(['Dense regions deconvolution', 'Compute clusters', 'Open results in Napari'], keys = ['do_dense_regions_deconvolution', 'do_cluster_computation', 'show_napari_corrector'], preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_Napari_correction], header= "Pipeline settings")
    
    event, values = prompt_with_help(layout_image_path, help= 'general', add_scrollbar=False)

    if event == 'Cancel' :
        return None

    im_path = values['image path']
    is_3D_stack = values['is_3D_stack']
    is_multichannel = values['is_multichannel']
    
    try :
        image = open_image(im_path)
        check_format(image, is_3D_stack, is_multichannel)
        values.update({'image' : image})
    except FormatError as error:
        sg.popup("Inconsistency between image format and options selected.\n Image shape : {0}".format(image.shape))
    except OSError as error :
        sg.popup('Image format not supported.')
    except ValueError as error :
        sg.popup('Image format not supported.')


    return values

def output_image_prompt(filename) :
    while True :
        relaunch = False
        layout = path_layout(['folder'], look_for_dir= True, header= "Output parameters :")
        layout += parameters_layout(["filename"], default_values= [filename + "_quantification"], size=25)
        layout += bool_layout(['csv','Excel', 'Feather'])
        layout.append([sg.Button('Cancel')])

        event,values= prompt(layout)

        values['filename'] = values['filename'].replace(".xlsx","")
        values['filename'] = values['filename'].replace(".feather","")
        excel_filename = values['filename'] + ".xlsx"
        feather_filename = values['filename'] + ".feather"

        if not values['Excel'] and not values['Feather'] and not values['csv'] :
            sg.popup("Please check at least one box : Excel/Feather/csv")
            relaunch = True
        elif not os.path.isdir(values['folder']) :
            sg.popup("Incorrect folder")
            relaunch = True
        elif os.path.isfile(values['folder'] + excel_filename) and values['Excel']:
            if ask_replace_file(excel_filename) :
                pass
            else :
                relaunch = True
        elif os.path.isfile(values['folder'] + feather_filename) and values['Feather']:
            if ask_replace_file(feather_filename) :
                pass
            else :
                relaunch = True

        if not relaunch : break

    if event == ('Cancel') : return None

    else : return values

def detection_parameters_promt(
        is_3D_stack, 
        is_multichannel, 
        do_dense_region_deconvolution, 
        do_clustering, 
        segmentation_done, 
        default_dict: dict
        ):
    """

    Returns Values
        
    """
    if is_3D_stack : dim = 3
    else : dim = 2

    #Detection
    detection_parameters = ['threshold', 'threshold penalty']
    default_detection = [default_dict.setdefault('threshold',''), default_dict.setdefault('threshold penalty', '1')]
    opt= [True, True]
    if is_multichannel : 
        detection_parameters += ['channel_to_compute']
        opt += [False]
        default_detection += [default_dict.setdefault('channel_to_compute', '')]
    layout = [[sg.Text("Green parameters", text_color= 'green'), sg.Text(" are optional parameters.")]]
    layout += parameters_layout(detection_parameters, header= 'Detection', opt=opt, default_values=default_detection)
    
    if dim == 2 : tuple_shape = ('y','x')
    else : tuple_shape = ('z','y','x')
    opt = {'voxel_size' : False, 'spot_size' : False, 'log_kernel_size' : True, 'minimum_distance' : True}
    unit = {'voxel_size' : 'nm', 'minimum_distance' : 'nm', 'spot_size' : 'radius(nm)', 'log_kernel_size' : 'px'}

    layout += tuple_layout(opt=opt, unit=unit, default_dict=default_dict, voxel_size= tuple_shape, spot_size= tuple_shape, log_kernel_size= tuple_shape, minimum_distance= tuple_shape)

    #Deconvolution
    if do_dense_region_deconvolution :
        default_dense_regions_deconvolution = [default_dict.setdefault('alpha',0.5), default_dict.setdefault('beta',1)]
        layout += parameters_layout(['alpha', 'beta',], default_values= default_dense_regions_deconvolution, header= 'do_dense_regions_deconvolution')
        layout += parameters_layout(['gamma'], unit= 'px', default_values= [default_dict.setdefault('gamma',5)])
        layout += tuple_layout(opt= {"deconvolution_kernel" : True}, unit= {"deconvolution_kernel" : 'px'}, default_dict=default_dict, deconvolution_kernel = tuple_shape)
    
    #Clustering
    if do_clustering :
        layout += parameters_layout(['cluster size'], unit="radius(nm)", default_values=[default_dict.setdefault('cluster size',400)])
        layout += parameters_layout(['min number of spots'], default_values=[default_dict.setdefault('min number of spots', 5)])

    if is_multichannel and segmentation_done :
        default_segmentation = [default_dict.setdefault('nucleus channel signal', default_dict.setdefault('nucleus channel',0))]
        layout += parameters_layout(['nucleus channel signal'], default_values=default_segmentation) + [[sg.Text(" channel from which signal will be measured for nucleus features.")]]

    layout += bool_layout(['Interactive threshold selector'], keys=['show_interactive_threshold_selector'], preset=[False])
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
        ['.csv','.excel','.feather'],
        keys= ['do_spots_csv', 'do_spots_excel', 'do_spots_feather'],
        preset= [default_dict.setdefault('do_spots_csv',False), default_dict.setdefault('do_spots_excel',False),default_dict.setdefault('do_spots_feather',False)]
    )

    event, values = prompt_with_help(layout, help='detection')
    if event == 'Cancel' : return None
    if is_3D_stack : values['dim'] = 3
    else : values['dim'] = 2
    return values


def ask_replace_file(filename:str) :
    layout = [
        [sg.Text("{0} already exists, replace ?")],
        [sg.Button('Yes'), sg.Button('No')]
    ]

    event, values = prompt(layout, add_ok_cancel= False)

    return event == 'Yes'

def ask_cancel_segmentation() :
    layout = [
        [sg.Text("Cancel segmentation ?")],
        [sg.Button('Yes'), sg.Button('No')]
    ]

    event, values = prompt(layout, add_ok_cancel= False)

    return event == 'Yes'

def ask_quit_small_fish() :
    layout = [
        [sg.Text("Quit small fish ?")],
        [sg.Button('Yes'), sg.Button('No')]
    ]

    event, values = prompt(layout, add_ok_cancel= False)

    return event == 'Yes'

def _error_popup(error:Exception) :
    sg.popup('Error : ' + str(error))
    raise error

def _warning_popup(warning:str) :
    sg.popup('Warning : ' + warning)

def _sumup_df(results: pd.DataFrame) :

    COLUMNS = ['acquisition_id','name','threshold', 'spot_number', 'cell_number', 'filename', 'channel_to_compute']

    if len(results) > 0 :
        if 'channel_to_compute' not in results : results['channel_to_compute'] = np.NaN
        res = results.loc[:,COLUMNS]
    else :
        res = pd.DataFrame(columns= COLUMNS)

    return res

def hub_prompt(fov_results, do_segmentation=False) -> 'Union[Literal["Add detection", "Compute colocalisation", "Batch detection", "Rename acquisition", "Save results", "Delete acquisitions", "Reset segmentation", "Reset results", "Segment cells"], dict[Literal["result_table", ""]]]':

    sumup_df = _sumup_df(fov_results)
    
    if do_segmentation :
        segmentation_object = sg.Text('Segmentation in memory', font='8', text_color= 'green')
    else :
        segmentation_object = sg.Text('No segmentation in memory', font='8', text_color= 'red')

    layout = [
        [sg.Text('RESULTS', font= 'bold 13')],
        [sg.Table(values= list(sumup_df.values), headings= list(sumup_df.columns), row_height=20, num_rows= 5, vertical_scroll_only=False, key= "result_table"), segmentation_object],
        [sg.Button('Segment cells'), sg.Button('Add detection'), sg.Button('Compute colocalisation'), sg.Button('Batch detection')],
        [sg.Button('Save results', button_color= 'green'), sg.Button('Save segmentation', button_color= 'green'), sg.Button('Load segmentation', button_color= 'green')],
        [sg.Button('Rename acquisition', button_color= 'gray'), sg.Button('Delete acquisitions',button_color= 'gray'), sg.Button('Reset segmentation',button_color= 'gray'), sg.Button('Reset all',button_color= 'gray')],
    ]

    window = sg.Window('small fish', layout= layout, margins= (10,10))

    while True : 
        event, values = window.read()
        if event == None : quit()
        elif event == 'Help' : pass
        else : 
            window.close()
            return event, values

def coloc_prompt() :
    layout = parameters_layout(['colocalisation distance'], unit= 'nm', header= 'Colocalisation', default_values= 0)
    event, values = prompt_with_help(layout)

    if event == 'Ok' :
        return values['colocalisation distance']
    else : return False

def rename_prompt() :
    layout = parameters_layout(['name'], header= "Rename acquisitions", size=12)
    event, values = prompt_with_help(layout)
    if event == 'Ok' :
        return values['name']
    else : return False

def ask_detection_confirmation(used_threshold) :
    layout = [
        [sg.Text("Proceed with current detection ?", font= 'bold 10')],
        [sg.Text("Threshold : {0}".format(used_threshold))],
        [sg.Button("Ok"), sg.Button("Restart detection")]
    ]

    event, value = prompt(layout, add_ok_cancel=False, add_scrollbar=False)

    if event == 'Restart detection' :
        return False
    else :
        return True
    
def ask_cancel_detection() :
    layout =[
        [sg.Text("Cancel new detection and return to main window ?", font= 'bold 10')],
        [sg.Button("Yes"), sg.Button("No")]
    ]

    event, value = prompt(layout, add_ok_cancel=False, add_scrollbar=False)

    if event == 'No' :
        return False
    else :
        return True


def ask_confirmation(question_displayed : str) :
    layout =[
        [sg.Text(question_displayed, font= 'bold 10')],
        [sg.Button("Yes"), sg.Button("No")]
    ]

    event, value = prompt(layout, add_ok_cancel=False, add_scrollbar=False)

    if event == 'No' :
        return False
    else :
        return True
    

def prompt_save_segmentation() -> 'dict[Literal["folder","filename","ext"]]':
    while True :
        relaunch = False
        layout = path_layout(['folder'], look_for_dir= True, header= "Output parameters :")
        layout += parameters_layout(["filename"], default_values= ["small_fish_segmentation"], size=25)
        layout += radio_layout(['npy','npz_uncompressed', 'npz_compressed'], key= 'ext')

        event,values= prompt(layout)
        if event == ('Cancel') : 
            return None

        values['filename'] = values['filename'].replace(".npy","")
        values['filename'] = values['filename'].replace(".npz","")
        filename = values['filename']

        if not os.path.isdir(values['folder']) :
            sg.popup("Incorrect folder")
            relaunch = True
        elif os.path.isfile(values['folder'] + filename):
            if ask_replace_file(filename) :
                pass
            else :
                relaunch = True

        if not relaunch : break

    return values

def prompt_load_segmentation() -> 'dict[Literal["nucleus","cytoplasm"]]':
    while True :
        relaunch = False
        layout = path_layout(['nucleus'], look_for_dir= False, header= "Load segmentation :")
        layout += path_layout(['cytoplasm'], look_for_dir= False)

        event,values= prompt(layout)
        if event == ('Cancel') : 
            return None
        
        if not os.path.isfile(values['nucleus']) :
            sg.popup("Incorrect nucleus file selected.")
            relaunch = True

        if not os.path.isfile(values['cytoplasm']) and values['cytoplasm'] != "" :
            sg.popup("Incorrect cytoplasm file selected.")
            relaunch = True
                

        if not relaunch : break


    return values