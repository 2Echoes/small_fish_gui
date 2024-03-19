from ..gui.prompts import output_image_prompt, post_analysis_prompt, _error_popup, prompt, prompt_with_help, input_image_prompt, detection_parameters_promt, ask_cancel_segmentation
from ..interface.output import save_results
from ._preprocess import check_integrity, convert_parameters_types
from .napari_wrapper import correct_spots
from .bigfish_wrappers import compute_snr_spots
import bigfish.plot as plot
import bigfish.detection as detection
import bigfish.stack as stack

import pandas as pd
import os
import numpy as np
import PySimpleGUI as sg
import small_fish.pipeline._segmentation as seg

def ask_input_parameters() :
    """
    Prompt user with interface allowing parameters setting for bigFish detection / deconvolution.
    
    Keys :
        - 'image path'
        - '3D stack'
        - 'time stack'
        - 'multichannel'
        - 'Dense regions deconvolution'
        - 'Segmentation
        - 'Napari correction'
        - 'threshold'
        - 'time step'
        - 'channel to compute'
        - 'alpha'
        - 'beta'
        - 'gamma'
        - 'voxel_size_{(z,y,x)}'
        - 'spot_size{(z,y,x)}'
        - 'log_kernel_size{(z,y,x)}'
        - 'minimum_distance{(z,y,x)}'
    """
    
    values = {}
    image_input_values = {}
    while True :
        is_3D_preset = image_input_values.setdefault('3D stack', False)
        is_time_preset = image_input_values.setdefault('time stack', False)
        is_multichannel_preset = image_input_values.setdefault('multichannel', False)
        denseregion_preset = image_input_values.setdefault('Dense regions deconvolution', False)
        do_segmentation_preset = image_input_values.setdefault('Segmentation', False)
        do_napari_preset = image_input_values.setdefault('Napari correction', False)

        image_input_values = input_image_prompt(
            is_3D_stack_preset=is_3D_preset,
            time_stack_preset=is_time_preset,
            multichannel_preset=is_multichannel_preset,
            do_dense_regions_deconvolution_preset=denseregion_preset,
            do_segmentation_preset=do_segmentation_preset,
            do_Napari_correction=do_napari_preset
        )

        if 'image' in image_input_values.keys() : 
            break
    values.update(image_input_values)
    values['dim'] = 3 if values['3D stack'] else 2
    if values['Segmentation'] and values['time stack'] : sg.popup('Segmentation is not supported for time stack. Segmentation will be turned off.')
    
    return values

def hub(image, voxel_size, spots_memory, results) :
    end_process = False
    next_action = post_analysis_prompt()
    try :
        if next_action == 'Save results' :
            dic = output_image_prompt()
            if isinstance(dic, dict) :
                path = dic['folder']
                filename = dic['filename']
                do_excel = dic['Excel']
                do_feather = dic['Feather']
                sucess = save_results(results, path= path, filename=filename, do_excel= do_excel, do_feather= do_feather)
                if sucess : sg.popup("Sucessfully saved at {0}.".format(path))

        elif next_action == 'add_detection' :
            user_parameters = initiate_detection()
            channel_to_compute = user_parameters.get('channel to compute')

            ##booleans
            is_time_stack = user_parameters['time stack']
            is_3D_stack = user_parameters['3D stack']
            multichannel = user_parameters['mutichannel']

            #image
            image_raw = user_parameters['image']
            images_gen = None #TODO to correct after modifications
            spots, result_frame = launch_detection(images_gen, user_parameters)

            if is_time_stack : spots_memory += [spots]
            else : spots_memory += spots
            results += [result_frame]
            end_process = False

        elif next_action == 'colocalisation' :
            pass

        elif next_action == 'open results in napari' :
            print("opening results in napari...")
            print("spots : ",type(spots_memory[0]))
            spots_memory = correct_spots(image, spots_memory, voxel_size)

        else :
            end_process = True

    except Exception as error :
        _error_popup(error)
    
    return image, voxel_size, spots_memory, results, end_process
    
def initiate_detection(is_3D_stack, is_time_stack, is_multichannel, do_dense_region_deconvolution) :
    user_parameters = detection_parameters_promt(is_3D_stack=is_3D_stack, is_time_stack=is_time_stack, is_multichannel=is_multichannel, do_dense_region_deconvolution=do_dense_region_deconvolution)
    user_parameters = convert_parameters_types(user_parameters)
    user_parameters = check_integrity(user_parameters, do_dense_region_deconvolution, is_time_stack, is_multichannel)

    return user_parameters


def launch_detection(image_input_values, images_gen) :
    
    #Extract parameters
    voxel_size = image_input_values['voxel_size']
    threshold = image_input_values.setdefault('threshold',None)
    spot_size = image_input_values.get('spot_size')
    log_kernel_size = image_input_values.get('log_kernel_size')
    minimum_distance = image_input_values.get('minimum_distance')
    use_napari =  image_input_values.setdefault('Napari correction', False)
    time_step = image_input_values.get('time step')

    ##deconvolution parameters
    do_dense_region_deconvolution = image_input_values['Dense regions deconvolution']
    alpha = image_input_values.get('alpha')
    beta = image_input_values.get('beta')
    gamma = image_input_values.get('gamma')
    deconvolution_kernel = image_input_values.get('deconvolution_kernel')

    ##Initiate lists
    spots_list = []
    
    res = {}
    #TODO 
    # for time stack auto threshold : compute threshold from random sample of frame -> pbwrap
    # Use meta data to pre fill voxel size and spot_size to 150 100 100
    # Add Help button with bigfish doc
    for step, image in enumerate(images_gen) :
        fov_res = {}
        
        #initial time is t = 0.
        print("Starting step {0}".format(step))
        if isinstance(time_step, (float, int)) :
            fov_res['time'] = time_step * step
        else : fov_res['time'] = np.NaN

        #detection
        spots, fov_res['threshold'] = detection.detect_spots(images= image, threshold=threshold, return_threshold= True, voxel_size=voxel_size, spot_radius= spot_size, log_kernel_size=log_kernel_size, minimum_distance=minimum_distance)
        
        if use_napari : spots = correct_spots(image, [spots], voxel_size)[0]
        if do_dense_region_deconvolution :
            spots, dense_regions, ref_spot = detection.decompose_dense(image=image, spots=spots, voxel_size=voxel_size, spot_radius=spot_size, kernel_size=deconvolution_kernel, alpha=alpha, beta=beta, gamma=gamma)

        #features
        fov_res['spot_number'] = len(spots)
        snr_res = compute_snr_spots(image, spots, voxel_size, spot_size)
        
        Z,Y,X = list(zip(*spots))
        spots_values = image[Z,Y,X]
        fov_res['spotsSignal_median'], fov_res['spotsSignal_mean'], fov_res['spotsSignal_std'] = np.median(spots_values), np.mean(spots_values), np.std(spots_values)
        fov_res['median_pixel'] = np.median(image)
        fov_res['mean_pixel'] = np.mean(image)

        #appending results
        fov_res.update(snr_res)

        for name, value in fov_res.items() :
            current_state = res.setdefault(name, [])
            res[name] = current_state + [value]

    result_frame = pd.DataFrame(res)
    
    return image, voxel_size, spots_list, result_frame

def launch_segmentation(image) :
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

    #Default parameters
    cyto_size = 30
    cytoplasm_channel = 0
    nucleus_size = 30
    nucleus_channel = 0
    path = os.getcwd()
    show_segmentation = False
    filename = 'cell_segmentation.png'
    available_channels = list(range(image.shape[0]))

    #Ask user for parameters
    #if incorrect parameters --> set relaunch to True
    while True :
        layout = seg._segmentation_layout(
            cytoplasm_channel_preset= cytoplasm_channel,
            nucleus_channel_preset= nucleus_channel,
            cyto_diameter_preset= cyto_size,
            nucleus_diameter_preset= nucleus_size,
            saving_path_preset= path,
            show_segmentation_preset=show_segmentation,
            filename_preset=filename
        )

        event, values = prompt_with_help(layout, help='segmentation')
        if event == 'Cancel' :
            cancel_segmentation = ask_cancel_segmentation()
        else : 
            cancel_segmentation = False
        
        if cancel_segmentation :
            return None, None

        #Extract parameters
        values = seg._cast_segmentation_parameters(values)
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
        if type(cyto_model_name) != str :
            sg.popup('Invalid cytoplasm model name.')
            relaunch= True
        if cytoplasm_channel not in available_channels :
            sg.popup('For given input image please select channel in {0}\ncytoplasm channel : {1}'.format(available_channels, cytoplasm_channel))
            relaunch= True
            cytoplasm_channel = 0

        if type(cyto_size) not in [int, float] :
            sg.popup("Incorrect cytoplasm size.")
            relaunch= True
            cyto_size = 30

        if type(nucleus_model_name) != str :
            sg.popup('Invalid nucleus model name.')
            relaunch= True
        if nucleus_channel not in available_channels :
            sg.popup('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
            relaunch= True
            nucleus_channel = 0
        if type(nucleus_size) not in [int, float] :
            sg.popup("Incorrect nucleus size.")
            relaunch= True
            nucleus_size = 30
        
        if not relaunch : break

    #Launching segmentation
    
    waiting_layout = [
        [sg.Text("Running segmentation...")]
    ]
    window = sg.Window(
        title= 'small_fish',
        layout= waiting_layout,
        grab_anywhere= True,
        no_titlebar= True
    )

    window.read(timeout= 30, close= False)

    try :
        if type(path) != type(None) and filename != '':
            output_path = path + '/' + filename
        else :
            output_path = None
        cytoplasm_label, nucleus_label = seg.cell_segmentation(
            image,
            cyto_model_name= cyto_model_name,
            cyto_diameter= cyto_size,
            nucleus_model_name= nucleus_model_name,
            nucleus_diameter= nucleus_size,
            channels=channels,
            )

    finally  : window.close()
    if show_segmentation or type(output_path) != type(None) :
        plot.plot_segmentation_boundary(stack.maximum_projection(image[cytoplasm_channel]), cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=show_segmentation, path_output=output_path)

    return cytoplasm_label, nucleus_label

