"""
Contains code to handle detection as well as bigfish wrappers related to spot detection.
"""

from ._preprocess import ParameterInputError
from ._preprocess import check_integrity, convert_parameters_types
from ._signaltonoise import compute_snr_spots
from ._detection_visualisation import correct_spots, _update_clusters
from ..gui import add_default_loading
from ..gui import detection_parameters_promt, input_image_prompt

import numpy as np
import pandas as pd
import PySimpleGUI as sg
import os
from numpy import NaN
import bigfish.detection as detection
import bigfish.stack as stack
import bigfish.multistack as multistack
import bigfish.classification as classification
from bigfish.detection.spot_detection import get_object_radius_pixel
from types import GeneratorType


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


def compute_auto_threshold(images, voxel_size=None, spot_radius=None, log_kernel_size=None, minimum_distance=None, im_number= 15, crop_zstack= None) :
    """
    Compute bigfish auto threshold efficiently for list of images. In case on large set of images user can set im_number to only consider a random subset of image for threshold computation.
    """
    # check parameters
    stack.check_parameter(images = (list, np.ndarray, GeneratorType,), voxel_size=(int, float, tuple, list, type(None)),spot_radius=(int, float, tuple, list, type(None)),log_kernel_size=(int, float, tuple, list, type(None)),minimum_distance=(int, float, tuple, list, type(None)), im_number = int, crop_zstack= (type(None), tuple))

    # if one image is provided we enlist it
    if not isinstance(images, list):
        if isinstance(images, np.ndarray) : 
            stack.check_array(images,ndim=[2, 3],dtype=[np.uint8, np.uint16, np.float32, np.float64])
            ndim = images.ndim
            images = [images]
        else : 
            images = [image for image in images]
            for image in images : stack.check_array(image,ndim=[2, 3],dtype=[np.uint8, np.uint16, np.float32, np.float64])
            ndim = images[0].ndim

    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(image,ndim=[2, 3],dtype=[np.uint8, np.uint16, np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
    if len(images) > im_number : #if true we select a random sample of images
        idx = np.arange(len(images),dtype= int)
        np.random.shuffle(idx)
        images = [images[i] for i in idx]
        
    #Building a giant 3D array containing all information for threshold selection -> cheating detection.automated_threshold_setting that doesn't take lists and doesn't use spatial information.
    if type(crop_zstack) == type(None) :
        crop_zstack = (0, len(images[0]))
    
    log_kernel_size, minimum_distance = _compute_threshold_parameters(ndim, voxel_size, spot_radius, minimum_distance, log_kernel_size)
    images_filtered = np.concatenate(
        [stack.log_filter(image[crop_zstack[0]: crop_zstack[1]], sigma= log_kernel_size) for image in images],
         axis= ndim -1)
    max_masks = np.concatenate(
        [detection.local_maximum_detection(image[crop_zstack[0]: crop_zstack[1]], min_distance= minimum_distance) for image in images],
         axis= ndim -1)
    threshold = detection.automated_threshold_setting(images_filtered, max_masks)

    return threshold

def _compute_threshold_parameters(ndim, voxel_size, spot_radius, minimum_distance, log_kernel_size) :

    # check consistency between parameters - detection with voxel size and
    # spot radius
    if (voxel_size is not None and spot_radius is not None
            and log_kernel_size is None and minimum_distance is None):
        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != ndim:
                raise ValueError("'voxel_size' must be a scalar or a sequence "
                                 "with {0} elements.".format(ndim))
        else:
            voxel_size = (voxel_size,) * ndim
        if isinstance(spot_radius, (tuple, list)):
            if len(spot_radius) != ndim:
                raise ValueError("'spot_radius' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            spot_radius = (spot_radius,) * ndim
        log_kernel_size = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        minimum_distance = get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)

    # check consistency between parameters - detection with kernel size and
    # minimal distance
    elif (voxel_size is None and spot_radius is None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError("'minimum_distance' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # check consistency between parameters - detection in priority with kernel
    # size and minimal distance
    elif (voxel_size is not None and spot_radius is not None
          and log_kernel_size is not None and minimum_distance is not None):
        if isinstance(log_kernel_size, (tuple, list)):
            if len(log_kernel_size) != ndim:
                raise ValueError("'log_kernel_size' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            log_kernel_size = (log_kernel_size,) * ndim
        if isinstance(minimum_distance, (tuple, list)):
            if len(minimum_distance) != ndim:
                raise ValueError("'minimum_distance' must be a scalar or a "
                                 "sequence with {0} elements.".format(ndim))
        else:
            minimum_distance = (minimum_distance,) * ndim

    # missing parameters
    else:
        raise ValueError("One of the two pairs of parameters ('voxel_size', "
                         "'spot_radius') or ('log_kernel_size', "
                         "'minimum_distance') should be provided.")
    
    return log_kernel_size, minimum_distance

def cluster_detection(spots, voxel_size, radius = 350, nb_min_spots = 4, keys_to_compute = ["clustered_spots", "clusters"]) :
    """
    Performs `bigfish.detection.cluster_detection()` to detect clusters.
    Then offers possibility to get results sorted in pandas dataframe.

    Parameters
    ----------
        spots : np.ndarray
            Coordinates of the detected spots with shape (nb_spots, 3) or (nb_spots, 2).
        voxel_size : int, float, Tuple(int, float) or List(int, float)
            Size of a voxel, in nanometer. One value per spatial dimension (zyx or yx dimensions). If it's a scalar, the same value is applied to every dimensions.
        radius : int
            The maximum distance between two samples for one to be considered as in the neighborhood of the other. Radius expressed in nanometer.
        nb_min_spots : int
            The number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). This includes the point itself.
        keys_to_compute : list[str], str
            keys from (clustered_spots, clusters, clustered_spots_dataframe, clusters_dataframe)
                --> clustered_spots : np.ndarray
                    Coordinates of the detected spots with shape (nb_spots, 4) or (nb_spots, 3). One coordinate per dimension (zyx or yx coordinates) plus the index of the cluster assigned to the spot. If no cluster was assigned, value is -1.
                --> clusters : np.ndarray
                    Array with shape (nb_clusters, 5) or (nb_clusters, 4). One coordinate per dimension for the clusters centroid (zyx or yx coordinates), the number of spots detected in the clusters and its index.
                --> clustered_spots_dataframe
                --> clusters_dataframe
    
    Returns
    -------
        res : dict
            keys : keys from `keys_to_compute` argument : (clustered_spots, clusters, clustered_spots_dataframe, clusters_dataframe)    
    """

    if isinstance(keys_to_compute, str) : keys_to_compute = [keys_to_compute]
    elif isinstance(keys_to_compute, list) : pass
    else : raise TypeError("Wrong type for keys_to_compute. Should be list[str] or str. It is {0}".format(type(keys_to_compute)))
    if len(spots) == 0 :
        res = {'clustered_spots' : [], 'clusters' : [], 'clustered_spots_dataframe' : pd.DataFrame(columns= ["id", "cluster_id", "z", "y", "x"]), 'clusters_dataframe' : pd.DataFrame(columns= ["id", "z", "y", "x", "spot_number"])}
        return {key : res[key] for key in keys_to_compute}
    else : res = {}
    voxel_size = tuple([int(d) for d in voxel_size])
    clustered_spots, clusters = detection.detect_clusters(spots, voxel_size= voxel_size, radius= radius, nb_min_spots= nb_min_spots)


    if 'clustered_spots' in keys_to_compute :
        res['clustered_spots'] = clustered_spots
        
    if 'clusters' in keys_to_compute : 
        res['clusters'] = clusters

    if 'clustered_spots_dataframe' in keys_to_compute :
        res['clustered_spots_dataframe'] = _compute_clustered_spots_dataframe(clustered_spots)
    
    if 'clusters_dataframe' in keys_to_compute :
        res['clusters_dataframe'] = _compute_cluster_dataframe(clusters)

    return res

def initiate_detection(user_parameters, segmentation_done, map, shape) :
    is_3D_stack= user_parameters['3D stack']
    is_multichannel = user_parameters['multichannel']
    do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
    do_clustering = user_parameters['Cluster computation']
    do_segmentation = user_parameters['Segmentation']
    
    while True :
        user_parameters = detection_parameters_promt(
            is_3D_stack=is_3D_stack,
            is_multichannel=is_multichannel,
            do_dense_region_deconvolution=do_dense_region_deconvolution,
            do_clustering=do_clustering,
            do_segmentation=do_segmentation,
            segmentation_done= segmentation_done,
            default_dict=user_parameters
            )
        
        if type(user_parameters) == type(None) : return user_parameters
        try :
            user_parameters = convert_parameters_types(user_parameters)
            user_parameters = check_integrity(user_parameters, do_dense_region_deconvolution, is_multichannel, segmentation_done, map, shape)
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
def launch_cell_extraction(acquisition_id, spots, clusters, image, nucleus_signal, cell_label, nucleus_label, user_parameters) :

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
    features_names = ['acquisition_id', 'cell_id', 'cell_bbox'] + classification.get_features_name(
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

        with np.errstate(divide= 'ignore', invalid= 'ignore') :
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
        features += [np.mean(nuc_signal), np.median(nuc_signal), np.max(nuc_signal), np.min(nuc_signal)]

        features = [acquisition_id, cell_id, cell_bbox] + features
        
        result_frame = pd.concat([
            result_frame,
            pd.DataFrame(columns = features_names, data= (features,)),
        ],
        axis= 0
        )
    
    return result_frame



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
        image,
        other_image,
        user_parameters,
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
    
    RETURNS
    -------
        user_parameters : dict
        spots : np.ndarray
        clusters : np.ndarray

    USER_PARAMETERS UPDATE
    ----------------------
        'threshold'
    """
    fov_result = {}
    do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
    do_clustering = user_parameters['Cluster computation']

    spots, threshold  = _launch_detection(image, user_parameters)
        
    if do_dense_region_deconvolution : 
        spots = launch_dense_region_deconvolution(image, spots, user_parameters)
        
    if do_clustering : 
        clusters = launch_clustering(spots, user_parameters) #012 are coordinates #3 is number of spots per cluster, #4 is cluster index
        clusters = _update_clusters(clusters, spots, voxel_size=user_parameters['voxel_size'], cluster_size=user_parameters['cluster size'], min_spot_number= user_parameters['min number of spots'], shape=image.shape)

    else : clusters = None

    spots, post_detection_dict = launch_post_detection(image, spots, user_parameters)
    user_parameters['threshold'] = threshold

    if user_parameters['Napari correction'] :

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
        
    fov_result.update(post_detection_dict)
    
    return user_parameters, fov_result, spots, clusters
            

def launch_features_computation(acquisition_id, image, nucleus_signal, spots, clusters, nucleus_label, cell_label, user_parameters, frame_results) :

    dim = image.ndim
            
    if user_parameters['Cluster computation'] : 
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
        
    return frame_results, cell_result_dframe

def _compute_clustered_spots_dataframe(clustered_spots) :
    if len(clustered_spots) == 0 : return pd.DataFrame(columns= ["id", "cluster_id", "z", "y", "x"])
    z, y ,x, cluster_index = list(zip(*clustered_spots))
    ids = np.arange(len(clustered_spots))

    df = pd.DataFrame({
        "id" : ids
        ,"cluster_id" : cluster_index
        ,"z" : z
        ,"y" : y
        ,"x" : x
    })

    null_idx = df[df['cluster_id'] == -1].index
    df.loc[null_idx, 'cluster_id'] = np.NaN

    return df

def _compute_cluster_dataframe(clusters) :
    if len(clusters) == 0 : return pd.DataFrame(columns= ["id", "z", "y", "x", "spot_number"])
    z, y, x, spots_number, cluster_index = list(zip(*clusters))

    df = pd.DataFrame({
        "id" : cluster_index
        ,"z" : z
        ,"y" : y
        ,"x" : x
        ,"spot_number" : spots_number
    })

    return df

def get_nucleus_signal(image, other_images, user_parameters) :
    if user_parameters['multichannel'] :
        rna_signal_channel = user_parameters['channel to compute']
        nucleus_signal_channel = user_parameters['nucleus channel signal']
        if type(nucleus_signal_channel) == type(None) :
            return np.zeros(shape=image.shape)

        if rna_signal_channel == nucleus_signal_channel :
            nucleus_signal == image
        
        elif nucleus_signal_channel > rna_signal_channel :
            nucleus_signal_channel -=1
            nucleus_signal = other_images[nucleus_signal_channel]
        
        elif nucleus_signal_channel < rna_signal_channel :
            nucleus_signal = other_images[nucleus_signal_channel]

        return nucleus_signal
    else :
        return image