import PySimpleGUI as sg
import pandas as pd
import os
from .layout import path_layout, parameters_layout, bool_layout, tuple_layout, combo_layout, add_header
from ..interface import open_image, check_format, FormatError
from .help_module import ask_help

def prompt(layout, add_ok_cancel=True, timeout=None, timeout_key='TIMEOUT_KEY') :
    """
    Default event : 'Ok', 'Cancel'
    """
    if add_ok_cancel : layout += [[sg.Button('Ok'), sg.Button('Cancel')]]

    
    window = sg.Window('small fish', layout=layout, margins=(10,10))
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

def prompt_with_help(layout, help =None) :
    layout += [[]]
    layout += [[sg.Button('Help')]]
    layout += [[sg.Button('Ok'), sg.Button('Cancel')]]
    
    window = sg.Window('small fish', layout=layout)
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
        time_stack_preset=False,
        multichannel_preset = False,
        do_dense_regions_deconvolution_preset= False,
        do_clustering_preset = False,
        do_segmentation_preset= False,
        do_Napari_correction= False,
        ask_for_segmentation= True
) :
    """
        Keys :
        - 'image path'
        - '3D stack'
        - 'time stack'
        - 'multichannel'
        - 'Dense regions deconvolution'
        - 'Segmentation'
        - 'Napari correction'

    Returns Values

    """
    layout_image_path = path_layout(['image path'], header= "Image")
    layout_image_path += bool_layout(['3D stack', 'time stack', 'multichannel'], preset= [is_3D_stack_preset, time_stack_preset, multichannel_preset])
    
    if ask_for_segmentation : 
        layout_image_path += bool_layout(['Dense regions deconvolution', 'Cluster computation', 'Segmentation', 'Napari correction'], preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_segmentation_preset, do_Napari_correction], header= "Pipeline settings")
    else : 
        layout_image_path += bool_layout(['Dense regions deconvolution', 'Cluster computation', 'Napari correction'], preset= [do_dense_regions_deconvolution_preset, do_clustering_preset, do_Napari_correction], header= "Pipeline settings")
    
    event, values = prompt_with_help(layout_image_path, help= 'general')

    if event == 'Cancel' :
        return None

    im_path = values['image path']
    is_3D_stack = values['3D stack']
    is_time_stack = values['time stack']
    is_multichannel = values['multichannel']
    if not ask_for_segmentation : values['Segmentation'] = False

    if is_time_stack :
        sg.popup("Sorry time stack images are not yet supported.")
        return values
    
    try :
        image = open_image(im_path)
        check_format(image, is_3D_stack, is_time_stack, is_multichannel)
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
        layout += bool_layout(['Excel', 'Feather'])
        layout.append([sg.Button('Cancel')])

        event,values= prompt(layout)

        values['filename'] = values['filename'].replace(".xlsx","")
        values['filename'] = values['filename'].replace(".feather","")
        excel_filename = values['filename'] + ".xlsx"
        feather_filename = values['filename'] + ".feather"

        if not values['Excel'] and not values['Feather'] :
            sg.popup("Please check at least one box : Excel/Feather")
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

def detection_parameters_promt(is_3D_stack, is_multichannel, do_dense_region_deconvolution, do_clustering, do_segmentation, segmentation_done, default_dict: dict) :
    """

    keys :
        - 'threshold'
        - 'threshold penalty
        - 'time step'
        - 'channel to compute'
        - 'alpha'
        - 'beta'
        - 'gamma'
        - 'voxel_size_{(z,y,x)}'
        - 'spot_size{(z,y,x)}'
        - 'log_kernel_size{(z,y,x)}'
        - 'minimum_distance{(z,y,x)}'
        - 'cluster size'
        - 'min number of spots'

    Returns Values
        
    """
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

    if (do_segmentation and is_multichannel) or (is_multichannel and segmentation_done):
        default_segmentation = [default_dict.setdefault('nucleus channel signal', default_dict.setdefault('nucleus channel',0))]
        layout += parameters_layout(['nucleus channel signal'], default_values=default_segmentation) + [[sg.Text(" channel from which signal will be measured for nucleus features.")]]

    event, values = prompt_with_help(layout, help='detection')
    if event == 'Cancel' : return None
    if is_3D_stack : values['dim'] = 3
    else : values['dim'] = 2
    return values

def post_analysis_prompt() :
    answer = events(['Save results','add_detection', 'colocalisation', 'open results in napari'])

    return answer

def events(event_list) :
    """
    Return event chose from user
    """
    
    layout = [
        [sg.Button(event) for event in event_list]
    ]

    event, values = prompt(layout, add_ok_cancel= False)
    return event

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

    if len(results) > 0 :
        res = results.loc[:,['acquisition_id', 'spot_number', 'cell_number', 'filename', 'channel to compute']]
    else :
        res = pd.DataFrame(columns= ['acquisition_id', 'spot_number', 'cell_number', 'filename', 'channel to compute'])

    return res

def hub_prompt(fov_results, do_segmentation=False) :

    sumup_df = _sumup_df(fov_results)
    
    if do_segmentation :
        segmentation_object = sg.Text('Segmentation was performed', font='8', text_color= 'green')
    else :
        segmentation_object = sg.Text('Segmentation was not performed', font='8', text_color= 'red')

    layout = [
        [sg.Text('RESULTS', font= 'bold 13')],
        [sg.Table(values= list(sumup_df.values), headings= list(sumup_df.columns), row_height=20, num_rows= 5, vertical_scroll_only=False, key= "result_table"), segmentation_object],
        [sg.Button('Add detection'), sg.Button('Compute colocalisation'), sg.Button('Batch detection')],
        # [sg.Button('Save results', button_color= 'green'), sg.Button('Delete acquisitions',button_color= 'gray'), sg.Button('Reset segmentation',button_color= 'gray'), sg.Button('Reset results',button_color= 'gray')]
        [sg.Button('Save results', button_color= 'green'), sg.Button('Reset results',button_color= 'gray')]
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
    layout = [
        [parameters_layout(['colocalisation distance'], header= 'Colocalisation', default_values= 0)]
    ]

    event, values = prompt_with_help(layout)

    if event == 'Ok' :
        return values['colocalisation distance']
    else : return False

def ask_detection_confirmation(used_threshold) :
    layout = [
        [sg.Text("Proceed with current detection ?", font= 'bold 10')],
        [sg.Text("Threshold : {0}".format(used_threshold))],
        [sg.Button("Ok"), sg.Button("Restart detection")]
    ]

    event, value = prompt(layout, add_ok_cancel=False)

    if event == 'Restart detection' :
        return False
    else :
        return True
    
def ask_cancel_detection() :
    layout =[
        [sg.Text("Cancel new detection and return to main window ?", font= 'bold 10')],
        [sg.Button("Yes"), sg.Button("No")]
    ]

    event, value = prompt(layout, add_ok_cancel=False)

    if event == 'No' :
        return False
    else :
        return True