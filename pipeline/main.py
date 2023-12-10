import bigfish.detection as detection
from ..gui import ask_input_parameters
from ..interface import open_image
from ._preprocess import prepare_image, check_integrity, convert_parameters_types
from . napari_wrapper import correct_spots, show_spots



user_parameters = ask_input_parameters()
user_parameters = convert_parameters_types(user_parameters)
user_parameters = check_integrity(user_parameters)


#Extract parameters
im_path = user_parameters['im_path']
voxel_size = user_parameters['voxel_size']
threshold = user_parameters.setdefault('threshold')
spot_size = user_parameters.get('spot_size')
log_kernel_size = user_parameters.get('log_kernel_size')
minimum_distance = user_parameters.get('minimum_distance')
use_napari =  user_parameters.setdefault('use_napari', False)
show_spots_res = user_parameters.setdefault('show_spots_res', False)
time_step = user_parameters.setdefault('time_step', 0)

do_dense_region_deconvolution = user_parameters['do_dense_region_deconvolution']
alpha = user_parameters.setdefault('alpha', 0.5)
beta = user_parameters.setdefault('beta', 1)
gamma = user_parameters.setdefault('gamma', 0)
deconvolution_kernel = user_parameters.get('deconvolution_kernel')

image = open_image(im_path) #TODO
integrity = check_integrity() #TODO
images_list = prepare_image(image)

for frame in range(0,len(images_list)) :
    image = images_list[frame]
    time = time_step * (frame + 1)
    
    #detection
    spots, threshold = detection.detect_spots(image= image, threshold=threshold, return_threshold= True, voxel_size=voxel_size, spot_radius= spot_size, log_kernel_size=log_kernel_size, minimum_distance=minimum_distance)
    if use_napari : spots = correct_spots(spots)
    spots = detection.decompose_dense(image=image, spots=spots, voxel_size=voxel_size, spot_radius=spot_size, kernel_size=deconvolution_kernel, alpha=alpha, beta=beta)
    if show_spots_res : show_spots(spots)

    #features
    spot_number = len(spots)
    snr = detection.compute_snr_spots()