
#Add keys hinting to user_parameters instance keys.

from typing import TypedDict, Tuple
from numpy import ndarray
    
class pipeline_parameters(TypedDict) :
            """
            At run time is a regular dict instance, this class is used for keys hinting
            """
            alpha : float
            beta : float
            channel_to_compute : int
            cluster_size : int
            do_cluster_computation : bool
            do_dense_regions_deconvolution : bool
            do_spots_excel : bool
            do_spots_feather : bool
            do_spots_csv : bool
            dim : int
            filename : str
            gamma : float
            image_path : str
            image : ndarray
            show_interactive_threshold_selector : bool
            log_kernel_size : Tuple[float,float,float]
            log_kernel_size_x : float
            log_kernel_size_y : float
            log_kernel_size_z : float
            min_number_of_spots : int
            minimum_distance : Tuple[float,float,float]
            minimum_distance_x : float
            minimum_distance_y : float
            minimum_distance_z : float
            is_3D_stack : bool
            is_multichannel : bool
            show_napari_corrector : bool
            nucleus_channel_signal : int
            reodered_shape : Tuple[int,int,int,int,int]
            do_segmentation : bool
            segmentation_done : bool
            shape : Tuple[int,int,int,int,int]
            spots_extraction_folder : str
            spots_filename : str
            spot_size : Tuple[int,int,int]
            spot_size_x : int
            spot_size_y : int
            spot_size_z : int
            threshold : int
            threshold_penalty : int
            time_stack : None
            time_step : None
            voxel_size : Tuple[float,float,float]
            voxel_size_x : float
            voxel_size_y : float
            voxel_size_z : float