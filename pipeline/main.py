import bigfish.detection as detection
import numpy as np
import pandas as pd
from ..gui import ask_input_parameters
from ._preprocess import prepare_image, check_integrity, convert_parameters_types
from . napari_wrapper import correct_spots, show_spots


#Open interface and ask user for parameters.
user_parameters = ask_input_parameters()
user_parameters = convert_parameters_types(user_parameters)
user_parameters = check_integrity(user_parameters)

#Extract parameters
im_path = user_parameters['image path']
voxel_size = user_parameters['voxel_size']
threshold = user_parameters.setdefault('threshold',None)
spot_size = user_parameters.get('spot_size')
log_kernel_size = user_parameters.get('log_kernel_size')
minimum_distance = user_parameters.get('minimum_distance')
use_napari =  user_parameters.setdefault('use_napari', False)
show_spots_res = user_parameters.setdefault('show_spots_res', False)
time_step = user_parameters.get('time_step')
channel_to_compute = user_parameters.get('channel to compute')


##One element per time step.

#Spots coordinates
spots_list = []
spot_number_list = []

#signal to noise ratio
##deconvolution parameters
do_dense_region_deconvolution = user_parameters['do_dense_region_deconvolution']
alpha = user_parameters.get('alpha')
beta = user_parameters.get('beta')
gamma = user_parameters.get('gamma')
deconvolution_kernel = user_parameters.get('deconvolution_kernel')

##booleans
is_time_stack = user_parameters['time stack']
is_3D_stack = user_parameters['3D stack']
multichannel = user_parameters['mutichannel']

#image
image_raw = user_parameters['image']
images_gen = prepare_image(image_raw, is_3D_stack, multichannel, is_time_stack)
snr_median_list = []
snr_mean_list = []
snr_std_list = []

#Signal detect at spots location
spotsSignal_median_list = []
spotsSignal_mean_list = []
spotsSignal_std_list = []

#pixel value from overall fov
median_pixel_list = []
mean_pixel_list = []


for step, image in enumerate(images_gen) :
    #initial time is t = 0.
    print("Starting step {0}".format(step))
    time = time_step * step

    #detection
    spots, threshold = detection.detect_spots(image= image, threshold=threshold, return_threshold= True, voxel_size=voxel_size, spot_radius= spot_size, log_kernel_size=log_kernel_size, minimum_distance=minimum_distance)
    if use_napari : spots = correct_spots(spots)
    spots = detection.decompose_dense(image=image, spots=spots, voxel_size=voxel_size, spot_radius=spot_size, kernel_size=deconvolution_kernel, alpha=alpha, beta=beta)
    if show_spots_res : show_spots(spots)

    #features
    spot_number = len(spots)
    snr = detection.compute_snr_spots()
    snr_median, snr_mean, snr_std = np.median(snr), np.mean(snr), np.std(snr)
    spots_values = image[spots]
    spotsSignal_median, spotsSignal_mean, spotsSignal_std = np.median(spots_values), np.mean(spots_values), np.std(spots_values)
    median_pixel = np.median(image)
    mean_pixel = np.mean(image)

    #appending results
    spots_list.append(spots)
    spot_number_list.append(spot_number)
    snr_median_list.append(snr_median)
    snr_mean_list.append(snr_mean)
    snr_std_list.append(snr_std)
    spotsSignal_median_list.append(spotsSignal_median)
    spotsSignal_mean_list.append(spotsSignal_mean)
    spotsSignal_std_list.append(spotsSignal_std)
    median_pixel_list.append(median_pixel)
    mean_pixel_list.append(mean_pixel)

result_frame = pd.DataFrame({
    'spot_number' : spot_number,
    'snr_meadian' : snr_median_list,
    'snr_mean' : snr_mean,
    'snr_std' : snr_std,
    'spot_median_signal' : spotsSignal_median_list,
    'spot_mean_signal' : spotsSignal_mean_list,
    'spot_std_signal' : spotsSignal_std_list,
    'median_siganl' : median_pixel_list,
    'mean_signal' : mean_pixel_list
})