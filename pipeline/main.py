from ._preprocess import prepare_image
from .actions import initiate_detection, launch_detection, hub

#Open interface and ask user for parameters.
user_parameters = initiate_detection()

#Extract parameters
channel_to_compute = user_parameters.get('channel to compute')

##booleans
is_time_stack = user_parameters['time stack']
is_3D_stack = user_parameters['3D stack']
multichannel = user_parameters['multichannel']

#image
image_raw = user_parameters['image']
images_gen = prepare_image(image_raw, is_3D_stack=is_3D_stack, multichannel=multichannel, is_time_stack=is_time_stack)

image, voxel_size, spots, result_frame = launch_detection(images_gen, user_parameters)
results= [result_frame]
spots_memory = [spots] if is_time_stack else spots
end_process = False

while not end_process :
    image, voxel_size, spots_memory, results, end_process = hub(image, voxel_size, spots_memory, results)