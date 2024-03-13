from ._preprocess import prepare_image_detection, reorder_image_stack, map_channels
from .actions import initiate_detection, launch_detection, hub, launch_segmentation, ask_input_parameters

#Open interface and ask user for parameters.
user_parameters = ask_input_parameters()

#Extract parameters
is_time_stack = user_parameters['time stack']
is_3D_stack = user_parameters['3D stack']
multichannel = user_parameters['multichannel']
do_segmentation = user_parameters['Segmentation'] and not is_time_stack
do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
image_raw = user_parameters['image']
map = map_channels(image_raw, is_3D_stack=is_3D_stack, is_time_stack=is_time_stack, multichannel=multichannel)

#Segmentation
if do_segmentation and not is_time_stack:
    im_seg = reorder_image_stack(map, image_raw)
    cytoplasm_label, nucleus_label = launch_segmentation(im_seg)

else :
    cytoplasm_label, nucleus_label = None,None

#Detection
user_parameters.update(initiate_detection(is_3D_stack, is_time_stack, multichannel, do_dense_region_deconvolution))
channel_to_compute = user_parameters.get('channel to compute')
images_gen = prepare_image_detection(map, image_raw)


image, voxel_size, spots, result_frame = launch_detection(user_parameters, images_gen)
results= [result_frame]
spots_memory = [spots] if is_time_stack else spots
end_process = False

while not end_process :
    image, voxel_size, spots_memory, results, end_process = hub(image, voxel_size, spots_memory, results)