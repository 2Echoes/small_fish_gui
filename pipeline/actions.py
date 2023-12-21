from ..gui.prompt import events, output_image_prompt, post_analysis_prompt, ask_input_parameters, _warning_popup, _error_popup
from ..interface.output import save_results
from ._preprocess import prepare_image, check_integrity, convert_parameters_types
from .napari_wrapper import correct_spots
from .bigfish_wrappers import compute_snr_spots
import bigfish.detection as detection

import pandas as pd
import numpy as np
import PySimpleGUI as sg


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
            images_gen = prepare_image(image_raw, is_3D_stack, multichannel, is_time_stack, channel_to_compute= channel_to_compute)
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
    
def initiate_detection() :
    user_parameters = ask_input_parameters()
    user_parameters = convert_parameters_types(user_parameters)
    user_parameters = check_integrity(user_parameters)

    return user_parameters

def launch_detection(images_gen, user_parameters: dict) :
    
    #Extract parameters
    voxel_size = user_parameters['voxel_size']
    threshold = user_parameters.setdefault('threshold',None)
    spot_size = user_parameters.get('spot_size')
    log_kernel_size = user_parameters.get('log_kernel_size')
    minimum_distance = user_parameters.get('minimum_distance')
    use_napari =  user_parameters.setdefault('Napari correction', False)
    time_step = user_parameters.get('time step')

    ##deconvolution parameters
    do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
    alpha = user_parameters.get('alpha')
    beta = user_parameters.get('beta')
    gamma = user_parameters.get('gamma')
    deconvolution_kernel = user_parameters.get('deconvolution_kernel')

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