from ..gui.prompts import output_image_prompt, _error_popup, prompt_with_help, input_image_prompt, detection_parameters_promt, ask_cancel_segmentation, hub_prompt, coloc_prompt, ask_detection_confirmation, prompt
from ..interface.output import save_results
from ..gui.animation import add_default_loading
from ._preprocess import check_integrity, convert_parameters_types, reorder_image_stack
from .napari_wrapper import correct_spots, _update_clusters
from .bigfish_wrappers import compute_snr_spots
from ._preprocess import ParameterInputError, map_channels, prepare_image_detection, reorder_shape
from ._detection import compute_auto_threshold, cluster_detection
from ._colocalisation import spots_multicolocalisation, spots_colocalisation
from ._custom_errors import MissMatchError
from numpy import NaN
import bigfish.plot as plot
import bigfish.detection as detection
import bigfish.stack as stack
import bigfish.multistack as multistack
import bigfish.classification as classification

import os
import pandas as pd
import numpy as np
import PySimpleGUI as sg
import small_fish.pipeline._segmentation as seg


def ask_input_parameters(ask_for_segmentation=True) :
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
        do_clustering_preset = image_input_values.setdefault('Cluster computation', False)
        do_segmentation_preset = image_input_values.setdefault('Segmentation', False)
        do_napari_preset = image_input_values.setdefault('Napari correction', False)

        image_input_values = input_image_prompt(
            is_3D_stack_preset=is_3D_preset,
            time_stack_preset=is_time_preset,
            multichannel_preset=is_multichannel_preset,
            do_dense_regions_deconvolution_preset=denseregion_preset,
            do_clustering_preset= do_clustering_preset,
            do_segmentation_preset=do_segmentation_preset,
            do_Napari_correction=do_napari_preset,
            ask_for_segmentation= ask_for_segmentation
        )
        if type(image_input_values) == type(None) :
            return image_input_values

        if 'image' in image_input_values.keys() :
            image_input_values['shape'] = image_input_values['image'].shape 
            break


    values.update(image_input_values)
    values['dim'] = 3 if values['3D stack'] else 2
    values['filename'] = os.path.basename(values['image path'])
    if values['Segmentation'] and values['time stack'] : sg.popup('Segmentation is not supported for time stack. Segmentation will be turned off.')
    
    return values

def hub(acquisition_id, results, cell_results, coloc_df, segmentation_done, user_parameters, cell_label, nucleus_label) :
    print(user_parameters.get('cyto_model_name'))
    event, values = hub_prompt(results, segmentation_done)
    try :
        if event == 'Save results' :
            if len(results) != 0 :
                dic = output_image_prompt(filename=results.iloc[0].at['filename'])

            else :
                dic = None 

            if isinstance(dic, dict) :
                path = dic['folder']
                filename = dic['filename']
                do_excel = dic['Excel']
                do_feather = dic['Feather']
                sucess1 = save_results(results, path= path, filename=filename, do_excel= do_excel, do_feather= do_feather)
                sucess2 = save_results(cell_results, path= path, filename=filename + '_cell_result', do_excel= do_excel, do_feather= do_feather)
                sucess3 = save_results(coloc_df, path= path, filename=filename + '_coloc_result', do_excel= do_excel, do_feather= do_feather)
                if sucess1 and sucess2 and sucess3 : sg.popup("Sucessfully saved at {0}.".format(path))
                else : sg.popup("Please check at least one box : Excel/Feather")

        elif event == 'Add detection' :

            if acquisition_id == -1 :
                ask_for_segmentation = True
            else : 
                ask_for_segmentation = False
            
            #Ask user
            user_parameters.update(ask_input_parameters(ask_for_segmentation= ask_for_segmentation))

            if type(user_parameters) == type(None) :
               return results, cell_results, coloc_df, acquisition_id
            
            #Extract parameters
            is_time_stack = user_parameters['time stack']
            is_3D_stack = user_parameters['3D stack']
            multichannel = user_parameters['multichannel']
            do_segmentation = user_parameters['Segmentation'] and not is_time_stack
            do_clustering = user_parameters['Cluster computation']
            do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
            image_raw = user_parameters['image']
            map = map_channels(image_raw, is_3D_stack=is_3D_stack, is_time_stack=is_time_stack, multichannel=multichannel)
            user_parameters['reordered_shape'] = reorder_shape(user_parameters['shape'], map)
            use_napari = user_parameters['Napari correction']
            
            if ask_for_segmentation and do_segmentation :
                #Segmentation
                if do_segmentation and not is_time_stack:
                    im_seg = reorder_image_stack(map, image_raw)
                    cytoplasm_label, nucleus_label, user_parameters = launch_segmentation(im_seg, user_parameters=user_parameters)

                else :
                    cytoplasm_label, nucleus_label = None,None

                if type(cytoplasm_label) == type(None) or type(nucleus_label) == type(None) :
                    do_segmentation = False

            #Detection preparation
            while True :
                detection_parameters = initiate_detection(is_3D_stack, is_time_stack, multichannel, do_dense_region_deconvolution, do_clustering, do_segmentation, user_parameters['segmentation_done'], map, image_raw.shape, user_parameters)

                if type(detection_parameters) != type(None) :
                    user_parameters.update(detection_parameters) 

                time_step = user_parameters.get('time step')
                use_napari = user_parameters['Napari correction']
                channel_to_compute = user_parameters.get('channel to compute')
                images_gen = prepare_image_detection(map, image_raw)

                image, nucleus_signal, user_parameters, spots, clusters, frame_results = launch_detection(
                    images_gen=images_gen,
                    user_parameters=user_parameters,
                    multichannel=multichannel,
                    channel_to_compute=channel_to_compute,
                    is_time_stack=is_time_stack,
                    time_step=time_step,
                    use_napari=use_napari,
                    cell_label=cell_label,
                    nucleus_label=nucleus_label
                )
                if use_napari:
                    if ask_detection_confirmation(user_parameters.get('threshold')) :
                        acquisition_id +=1
                        break
                else :
                    acquisition_id += 1
                    break

            #Detection
            res, cell_res = launch_features_computation(
            acquisition_id=acquisition_id,
            image=image,
            nucleus_signal = nucleus_signal,
            dim=image.ndim,
            spots=spots,
            clusters=clusters,
            nucleus_label = nucleus_label,
            cell_label= cell_label,
            user_parameters=user_parameters,
            frame_results=frame_results,
            do_clustering=do_clustering
            )

            results = pd.concat([results, res])
            cell_results = pd.concat([cell_results, cell_res])
            
        elif event == 'Compute colocalisation' :
            print('Compute colocalisation')
            result_tables = values.setdefault('result_table', [])
            colocalisation_distance = initiate_colocalisation(result_tables)

            if colocalisation_distance == False :
                pass
            else :
                res_coloc = launch_colocalisation(result_tables, result_dataframe=results, colocalisation_distance=colocalisation_distance)

                coloc_df = pd.concat([
                    coloc_df,
                    res_coloc
                ],
                axis= 0)
        
        elif event == "Reset results" :
            print("restart")
            results = pd.DataFrame()
            cell_results = pd.DataFrame()
            coloc_df = pd.DataFrame()
            acquisition_id = -1
            user_parameters['segmentation_done'] = False

    except Exception as error :
        sg.popup(str(error))

    
    return results, cell_results, coloc_df, acquisition_id, user_parameters
    
def initiate_detection(is_3D_stack, is_time_stack, is_multichannel, do_dense_region_deconvolution, do_clustering, do_segmentation, segmentation_done, map, shape, default_dict={}) :
    while True :
        user_parameters = detection_parameters_promt(
            is_3D_stack=is_3D_stack,
            is_time_stack=is_time_stack,
            is_multichannel=is_multichannel,
            do_dense_region_deconvolution=do_dense_region_deconvolution,
            do_clustering=do_clustering,
            do_segmentation=do_segmentation,
            segmentation_done= segmentation_done,
            default_dict=default_dict
            )
        
        if type(user_parameters) == type(None) : return user_parameters
        try :
            user_parameters = convert_parameters_types(user_parameters)
            user_parameters = check_integrity(user_parameters, do_dense_region_deconvolution, is_time_stack, is_multichannel, segmentation_done, map, shape)
        except ParameterInputError as error:
            sg.popup(error)
        else :
            break
    return user_parameters

@add_default_loading
def _launch_detection(image, image_input_values: dict, time_stack_gen=None) :

    """
    Performs spots detection
    """
    
    #Extract parameters
    voxel_size = image_input_values['voxel_size']
    threshold = image_input_values.get('threshold')
    threshold_penalty = image_input_values.setdefault('threshold penalty', 1)
    print('threshold penalty : ', threshold_penalty)
    spot_size = image_input_values.get('spot_size')
    log_kernel_size = image_input_values.get('log_kernel_size')
    minimum_distance = image_input_values.get('minimum_distance')

    if type(threshold) == type(None) : 
        #detection
        if type(time_stack_gen) != type(None) :
            image_sample = time_stack_gen()
        else :
            image_sample = image
    
        threshold = compute_auto_threshold(image_sample, voxel_size=voxel_size, spot_radius=spot_size) * threshold_penalty
        print("auto threshold result : ", threshold)
    
    spots = detection.detect_spots(
        images= image,
        threshold=threshold,
        return_threshold= False,
        voxel_size=voxel_size,
        spot_radius= spot_size,
        log_kernel_size=log_kernel_size,
        minimum_distance=minimum_distance
        )
        
    return spots, threshold

@add_default_loading
def launch_dense_region_deconvolution(image, spots, image_input_values: dict,) :
    """
    Performs spot decomposition

    Returns
    -------
        spots : np.ndarray
            Array(nb_spot, dim) (dim is either 3 or 2)
        fov_res : dict
            keys : spot_number, spotsSignal_median, spotsSignal_mean, spotsSignal_std, median_pixel, mean_pixel, snr_median, snr_mean, snr_std, cell_medianbackground_mean, cell_medianbackground_std, cell_meanbackground_mean, cell_meanbackground_std, cell_stdbackground_mean, cell_stdbackground_std
    """
    
    ##Initiate lists
    voxel_size = image_input_values['voxel_size']
    spot_size = image_input_values.get('spot_size')
    ##deconvolution parameters
    alpha = image_input_values.get('alpha')
    beta = image_input_values.get('beta')
    gamma = image_input_values.get('gamma')
    deconvolution_kernel = image_input_values.get('deconvolution_kernel')
    dim = image_input_values['dim']
        
    spots, dense_regions, ref_spot = detection.decompose_dense(image=image, spots=spots, voxel_size=voxel_size, spot_radius=spot_size, kernel_size=deconvolution_kernel, alpha=alpha, beta=beta, gamma=gamma)
    del dense_regions, ref_spot

    return spots

@add_default_loading
def launch_post_detection(image, spots, image_input_values: dict,) :
    fov_res = {}
    dim = image_input_values['dim']
    voxel_size = image_input_values['voxel_size']
    spot_size = image_input_values.get('spot_size')

    #features
    fov_res['spot_number'] = len(spots)
    snr_res = compute_snr_spots(image, spots, voxel_size, spot_size)
        
    if dim == 3 :
        Z,Y,X = list(zip(*spots))
        spots_values = image[Z,Y,X]
    else :
        Y,X = list(zip(*spots))
        spots_values = image[Y,X]

    fov_res['spotsSignal_median'], fov_res['spotsSignal_mean'], fov_res['spotsSignal_std'] = np.median(spots_values), np.mean(spots_values), np.std(spots_values)
    fov_res['median_pixel'] = np.median(image)
    fov_res['mean_pixel'] = np.mean(image)

    #appending results
    fov_res.update(snr_res)

    return spots, fov_res

@add_default_loading
def launch_cell_extraction(acquisition_id, spots, clusters, image, nucleus_signal, cell_label, nucleus_label, user_parameters, time_stamp,) :

    #Extract parameters
    dim = user_parameters['dim']
    do_clustering = user_parameters['Cluster computation']
    voxel_size = user_parameters['voxel_size']

    if do_clustering : other_coords = {'clusters_coords' : clusters} if len(clusters) > 0 else None
    else : other_coords = None
    if do_clustering : do_clustering = len(clusters) > 0

    if image.ndim == 3 :
        image = stack.maximum_projection(image)
    if nucleus_signal.ndim == 3 :
        nucleus_signal = stack.maximum_projection(nucleus_signal)
    
    cells_results = multistack.extract_cell(
        cell_label=cell_label,
        ndim=dim,
        nuc_label=nucleus_label,
        rna_coord=spots,
        others_coord=other_coords,
        image=image
    )

    #BigFish features
    features_names = ['acquisition_id', 'cell_id', 'time', 'cell_bbox'] + classification.get_features_name(
        names_features_centrosome=False,
        names_features_area=True,
        names_features_dispersion=True,
        names_features_distance=True,
        names_features_foci=do_clustering,
        names_features_intranuclear=True,
        names_features_protrusion=False,
        names_features_topography=True
    )

    #Nucleus features : area is computed in bigfish
    features_names += ['nucleus_mean_signal', 'nucleus_median_signal', 'nucleus_max_signal', 'nucleus_min_signal']

    result_frame = pd.DataFrame()

    for cell in cells_results :

        #Extract cell results
        cell_id = cell['cell_id']
        cell_mask = cell['cell_mask']
        nuc_mask = cell ['nuc_mask']
        cell_bbox = cell['bbox'] # (min_y, min_x, max_y, max_x)
        min_y, min_x, max_y, max_x = cell['bbox'] # (min_y, min_x, max_y, max_x)
        nuc_signal = nucleus_signal[min_y:max_y, min_x:max_x]
        rna_coords = cell['rna_coord']
        foci_coords = cell.get('clusters_coords')
        signal = cell['image']

        features = classification.compute_features(
            cell_mask=cell_mask,
            nuc_mask=nuc_mask,
            ndim=dim,
            rna_coord= rna_coords,
            foci_coord=foci_coords,
            voxel_size_yx= float(voxel_size[-1]),
            smfish=signal,
            centrosome_coord=None,
            compute_centrosome=False,
            compute_area=True,
            compute_dispersion=True,
            compute_distance=True,
            compute_foci= do_clustering and len(clusters) > 0,
            compute_intranuclear=True,
            compute_protrusion=False,
            compute_topography=True
        )

        features = list(features)
        print("nuc signal shape : ", nuc_signal.shape)
        features += [np.mean(nuc_signal), np.median(nuc_signal), np.max(nuc_signal), np.min(nuc_signal)]

        features = [acquisition_id, cell_id, time_stamp, cell_bbox] + features
        
        result_frame = pd.concat([
            result_frame,
            pd.DataFrame(columns = features_names, data= (features,)),
        ],
        axis= 0
        )
    
    return result_frame

def launch_segmentation(image: np.ndarray, user_parameters: dict) :
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

    while True : # Loop if show_segmentation 
        #Default parameters
        cyto_model_name = user_parameters.setdefault('cyto_model_name', 'cyto2')
        cyto_size = user_parameters.setdefault('cyto size', 180)
        cytoplasm_channel = user_parameters.setdefault('cytoplasm channel', 0)
        nucleus_model_name = user_parameters.setdefault('nucleus_model_name', 'nuclei')
        nucleus_size = user_parameters.setdefault('nucleus size', 130)
        nucleus_channel = user_parameters.setdefault('nucleus channel', 0)
        path = os.getcwd()
        show_segmentation = False
        filename = user_parameters['filename'] + '_cell_segmentation.png'
        available_channels = list(range(image.shape[0]))

        print("Before prompt : ", cyto_model_name)
        print("Before prompt : ", nucleus_channel)

    #Ask user for parameters
    #if incorrect parameters --> set relaunch to True
        while True :
            layout = seg._segmentation_layout(
                cytoplasm_model_preset = cyto_model_name,
                cytoplasm_channel_preset= cytoplasm_channel,
                nucleus_model_preset = nucleus_model_name,
                nucleus_channel_preset= nucleus_channel,
                cyto_diameter_preset= cyto_size,
                nucleus_diameter_preset= nucleus_size,
                saving_path_preset= path,
                show_segmentation_preset=show_segmentation,
                filename_preset=filename,
            )

            event, values = prompt_with_help(layout, help='segmentation')
            if event == 'Cancel' :
                cancel_segmentation = ask_cancel_segmentation()

                if cancel_segmentation :
                    return None, None
                else : 
                    continue

            #Extract parameters
            values = seg._cast_segmentation_parameters(values)
            do_only_nuc = values['Segment only nuclei']
            cyto_model_name = values['cyto_model_name']
            cyto_size = values['cytoplasm diameter']
            cytoplasm_channel = values['cytoplasm channel']
            nucleus_model_name = values['nucleus_model_name']
            nucleus_size = values['nucleus diameter']
            nucleus_channel = values['nucleus channel']
            print("after prompt : nucleus_channel", values.get('nucleus channel'))
            path = values['saving path'] if values['saving path'] != '' else None
            show_segmentation = values['show segmentation']
            filename = values['filename'] if type(path) != type(None) else None
            channels = [cytoplasm_channel, nucleus_channel]

            relaunch= False
            #Checking integrity of parameters
            if type(cyto_model_name) != str  and not do_only_nuc:
                sg.popup('Invalid cytoplasm model name.')
                cyto_model_name = user_parameters.setdefault('cyto_model_name', 'cyto2')
                relaunch= True
            if cytoplasm_channel not in available_channels and not do_only_nuc:
                sg.popup('For given input image please select channel in {0}\ncytoplasm channel : {1}'.format(available_channels, cytoplasm_channel))
                relaunch= True
                cytoplasm_channel = user_parameters.setdefault('cytoplasm_channel',0)

            if type(cyto_size) not in [int, float] and not do_only_nuc:
                sg.popup("Incorrect cytoplasm size.")
                relaunch= True
                cyto_size = user_parameters.setdefault('cyto_size', 30)

            if type(nucleus_model_name) != str :
                sg.popup('Invalid nucleus model name.')
                nucleus_model_name = user_parameters.setdefault('nucleus_model_name', 'nuclei')
                relaunch= True
            if nucleus_channel not in available_channels :
                sg.popup('For given input image please select channel in {0}\nnucleus channel : {1}'.format(available_channels, nucleus_channel))
                relaunch= True
                nucleus_channel = user_parameters.setdefault('nucleus_channel', 0)
            if type(nucleus_size) not in [int, float] :
                sg.popup("Incorrect nucleus size.")
                relaunch= True
                nucleus_size = user_parameters.setdefault('nucleus_size', 30)

            if not relaunch : break

        #Launching segmentation
        waiting_layout = [
            [sg.Text("Running segmentation...")]
        ]
        window = sg.Window(
            title= 'small_fish',
            layout= waiting_layout,
            grab_anywhere= True,
            no_titlebar= False
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
                do_only_nuc=do_only_nuc
                )

        finally  : window.close()
        if show_segmentation or type(output_path) != type(None) :
            if do_only_nuc : im_proj = image[nucleus_channel]
            else : im_proj = image[cytoplasm_channel]
            if im_proj.ndim == 3 :
                im_proj = stack.maximum_projection(im_proj)
            plot.plot_segmentation_boundary(im_proj, cytoplasm_label, nucleus_label, boundary_size=2, contrast=True, show=show_segmentation, path_output=output_path)
        if show_segmentation :
            layout = [
                [sg.Text("Proceed with current segmentation ?")],
                [sg.Button("Yes"), sg.Button("No")]
            ]
            
            event, values = prompt(layout=layout, add_ok_cancel=False)
            if event == "Yes" :
                break
        else :
            break


    user_parameters.update(values)
    print("after segmentation : ", user_parameters.get('cyto_model_name'))
    print("after segmentation : ", user_parameters.get('cyto_model_name'))
    print("after segmentation : ", user_parameters.get('nucleus_channel'))
    return cytoplasm_label, nucleus_label, user_parameters

@add_default_loading
def launch_clustering(spots, user_parameters): 

    voxel_size = user_parameters['voxel_size']
    nb_min_spots = user_parameters['min number of spots']
    cluster_size = user_parameters['cluster size']

    clusters = cluster_detection(
        spots=spots,
        voxel_size=voxel_size,
        radius=cluster_size,
        nb_min_spots=nb_min_spots,
        keys_to_compute= 'clusters'
    )['clusters']

    return clusters

def launch_detection(
        images_gen,
        user_parameters,
        multichannel,
        channel_to_compute,
        is_time_stack,
        time_step,
        use_napari,
        cell_label= None,
        nucleus_label = None
        ) :
    """
    Main call for features computation :
    --> spot dection
    --> dense regions deconv
    --> cluster (opt)
    --> spot correction
    --> general features computations
    --> cell extractions
    --> cell features computations
    
    """
    
    do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
    do_clustering = user_parameters['Cluster computation']

    for step, image in enumerate(images_gen) :    
        frame_results = {}
        if is_time_stack :
            time = step * user_parameters['time step']
        else : time = NaN

        if multichannel : 
            print(channel_to_compute)
            print("type : ", type(channel_to_compute))
            print(image.shape)
            nucleus_signal_channel = user_parameters.get('nucleus channel signal')
            nucleus_signal = image[nucleus_signal_channel]
            other_image = [image[c] for c in range(image.shape[0])]
            del other_image[channel_to_compute]
            image = image[channel_to_compute]
        else : 
            other_image = []
            nucleus_signal = image

        if is_time_stack : #initial time is t = 0.
            print("Starting step {0}".format(step))
            if isinstance(time_step, (float, int)) :
                frame_results['time'] = time_step * step
            else : frame_results['time'] = NaN
        else : frame_results['time'] = NaN

        spots, threshold  = _launch_detection(image, user_parameters)
        
        if do_dense_region_deconvolution : 
            spots = launch_dense_region_deconvolution(image, spots, user_parameters)
        
        if do_clustering : 
            clusters = launch_clustering(spots, user_parameters) #012 are coordinates #3 is number of spots per cluster, #4 is cluster index
            clusters = _update_clusters(clusters, spots, voxel_size=user_parameters['voxel_size'], cluster_size=user_parameters['cluster size'], min_spot_number= user_parameters['min number of spots'], shape=image.shape)

        else : clusters = None

        spots, post_detection_dict = launch_post_detection(image, spots, user_parameters)
        post_detection_dict['threshold'] = threshold

        if use_napari : 
            spots, clusters = correct_spots(
                image, 
                spots, 
                user_parameters['voxel_size'],
                clusters=clusters,
                cluster_size= user_parameters.get('cluster size'),
                min_spot_number= user_parameters.setdefault('min number of spots', 0),
                cell_label=cell_label,
                nucleus_label=nucleus_label,
                other_images=other_image
                )
        
        user_parameters.update(post_detection_dict)
    
    return image, nucleus_signal, user_parameters, spots, clusters, frame_results
            

def launch_features_computation(acquisition_id, image, nucleus_signal, dim, spots, clusters, nucleus_label, cell_label, user_parameters, frame_results, do_clustering) :

    dim = user_parameters['dim']
    result = pd.DataFrame()
    result_cell_frame = pd.DataFrame()
            
    if do_clustering : 
        frame_results['cluster_number'] = len(clusters)
        if dim == 3 :
            frame_results['total_spots_in_clusters'] = clusters.sum(axis=0)[3]
        else :
            frame_results['total_spots_in_clusters'] = clusters.sum(axis=0)[2]
    
    if type(cell_label) != type(None) and type(nucleus_label) != type(None): 
        cell_result_dframe = launch_cell_extraction(
            acquisition_id=acquisition_id,
            spots=spots,
            clusters=clusters,
            image=image,
            nucleus_signal=nucleus_signal,
            cell_label= cell_label,
            nucleus_label=nucleus_label,
            user_parameters=user_parameters,
            time_stamp=user_parameters.get('time')
        )
    else :
        cell_result_dframe = pd.DataFrame()

    frame_results['acquisition_id'] = acquisition_id
    if type(cell_label) != type(None) and type(nucleus_label) != type(None):
        frame_results['cell_number'] = len(cell_result_dframe) 
    else : 
        frame_results['cell_number'] = NaN
    frame_results['spots'] = spots
    frame_results['clusters'] = clusters
    frame_results.update(user_parameters)
    frame_results['threshold'] = user_parameters['threshold']

    frame_results = pd.DataFrame(columns= frame_results.keys(), data= (frame_results.values(),))

    result: pd.DataFrame = pd.concat([result, frame_results])
    result_cell_frame: pd.DataFrame = pd.concat([result_cell_frame, cell_result_dframe])
        
    return result, result_cell_frame

def initiate_colocalisation(result_tables) :
    if len(result_tables) != 2 : 
        sg.popup("Please select 2 acquisitions for colocalisation (Ctrl + click in the table)")
        return False
    else : 
        while True :
            colocalisation_distance = coloc_prompt()
            if colocalisation_distance == False : return False
            try : 
                colocalisation_distance = int(colocalisation_distance)
            except Exception :
                sg.popup("Incorrect colocalisation distance")
            else :
                break
        return colocalisation_distance

@add_default_loading
def launch_colocalisation(result_tables, result_dataframe, colocalisation_distance) :
    """

    Target :

    - acquisition_couple
    - colocalisation_distance
    - spot1_total
    - spot2_total
    - fraction_spot1_coloc_spots
    - fraction_spot2_coloc_spots
    - fraction_spot1_coloc_clusters
    - fraction_spot2_coloc_spots

    """

    acquisition1 = result_dataframe.iloc[result_tables[0]]
    acquisition2 = result_dataframe.iloc[result_tables[1]]

    voxel_size1 = acquisition1.at['voxel_size']
    voxel_size2 = acquisition2.at['voxel_size']
    shape1 = acquisition1.at['reordered_shape']
    shape2 = acquisition2.at['reordered_shape']

    if voxel_size1 != voxel_size2 : 
        raise MissMatchError("voxel size 1 different than voxel size 2")
    else :
        voxel_size = voxel_size1

    if shape1 != shape2 :
        print(shape1)
        print(shape2) 
        raise MissMatchError("shape 1 different than shape 2")
    else :
        shape = shape1
        print(shape1)
        print(shape2)

    acquisition_couple = (acquisition1.at['acquisition_id'], acquisition2.at['acquisition_id'])

    spots1 = acquisition1['spots']
    spots2 = acquisition2['spots']

    spot1_total = len(spots1)
    spot2_total = len(spots2)

    try :
        fraction_spots1_coloc_spots2 = spots_colocalisation(image_shape=shape, spot_list1=spots1, spot_list2=spots2, distance= colocalisation_distance, voxel_size=voxel_size) / spot1_total
        fraction_spots2_coloc_spots1 = spots_colocalisation(image_shape=shape, spot_list1=spots2, spot_list2=spots1, distance= colocalisation_distance, voxel_size=voxel_size) / spot2_total
    except MissMatchError as e :
        sg.popup(str(e))
        fraction_spots1_coloc_spots2 = NaN
        fraction_spots2_coloc_spots1 = NaN

    if 'clusters' in acquisition1.index :
        try : 
            clusters1 = acquisition1['clusters'][:,:len(voxel_size)]
            fraction_spots2_coloc_cluster1 = spots_colocalisation(image_shape=shape, spot_list1=spots2, spot_list2=clusters1, distance= colocalisation_distance, voxel_size=voxel_size) / spot2_total
        except MissMatchError as e :
            sg.popup(str(e))
            fraction_spots2_coloc_cluster1 = NaN

    else : fraction_spots2_coloc_cluster1 = NaN

    if 'clusters' in acquisition2.index :
        try :
            clusters2 = acquisition2['clusters'][:,:len(voxel_size)]
            fraction_spots1_coloc_cluster2 = spots_colocalisation(image_shape=shape, spot_list1=spots1, spot_list2=clusters2, distance= colocalisation_distance, voxel_size=voxel_size) / spot1_total
        except MissMatchError as e :
            sg.popup(str(e))
            fraction_spots1_coloc_cluster2 = NaN

    else : fraction_spots1_coloc_cluster2 = NaN

    coloc_df = pd.DataFrame({
        "acquisition_couple" : [acquisition_couple],
        "acquisition_id_1" : [acquisition_couple[0]],
        "acquisition_id_2" : [acquisition_couple[1]],
        "colocalisation_distance" : [colocalisation_distance],
        "spot1_total" : [spot1_total],
        "spot2_total" : [spot2_total],
        'fraction_spots1_coloc_spots2' : [fraction_spots1_coloc_spots2],
        'fraction_spots2_coloc_spots1' : [fraction_spots2_coloc_spots1],
        'fraction_spots2_coloc_cluster1' : [fraction_spots2_coloc_cluster1],
        'fraction_spots1_coloc_cluster2' : [fraction_spots1_coloc_cluster2],
    })

    print(coloc_df.loc[:,['fraction_spots1_coloc_spots2','fraction_spots2_coloc_spots1', 'fraction_spots2_coloc_cluster1', 'fraction_spots1_coloc_cluster2']])

    return coloc_df