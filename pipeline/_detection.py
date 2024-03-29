import bigfish.stack as stack
import bigfish.detection as detection
import numpy as np
import pandas as pd

from types import GeneratorType
from bigfish.detection.spot_detection import get_object_radius_pixel


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