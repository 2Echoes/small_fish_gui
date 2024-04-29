from ._preprocess import prepare_image_detection, reorder_image_stack, map_channels, reorder_shape
from .actions import initiate_detection, hub, launch_segmentation, ask_input_parameters, launch_features_computation, launch_detection
from ..gui import ask_detection_confirmation
import pandas as pd

#Open interface and ask user for parameters.
user_parameters = ask_input_parameters()
if type(user_parameters) == type(None) : quit()
acquisition_id= 0

#Extract parameters
is_time_stack = user_parameters.setdefault('time stack', False)
is_3D_stack = user_parameters['3D stack']
multichannel = user_parameters['multichannel']
do_segmentation = user_parameters['Segmentation'] and not is_time_stack
do_dense_region_deconvolution = user_parameters['Dense regions deconvolution']
do_clustering = user_parameters['Cluster computation']
use_napari = user_parameters['Napari correction']
image_raw = user_parameters['image']
map = map_channels(image_raw, is_3D_stack=is_3D_stack, is_time_stack=is_time_stack, multichannel=multichannel)
user_parameters['reordered_shape'] = reorder_shape(user_parameters['shape'], map)

#Segmentation
if do_segmentation and not is_time_stack:
    im_seg = reorder_image_stack(map, image_raw)
    cytoplasm_label, nucleus_label, user_parameters = launch_segmentation(im_seg, user_parameters=user_parameters)

else :
    cytoplasm_label, nucleus_label = None,None

if type(cytoplasm_label) == type(None) or type(nucleus_label) == type(None) :
    do_segmentation = False
    user_parameters['segmentation_done'] = False

else : user_parameters['segmentation_done'] = True

#Detection

while True :
    detection_parameters = initiate_detection(is_3D_stack, is_time_stack, multichannel, do_dense_region_deconvolution, do_clustering, do_segmentation, user_parameters['segmentation_done'], map, image_raw.shape, user_parameters)

    if type(detection_parameters) != type(None) :
        user_parameters.update(detection_parameters) 
    else : #If user click cancel will close small fish
        quit()

    time_step = user_parameters.get('time step')
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
        cell_label=cytoplasm_label,
        nucleus_label=nucleus_label
    )
    if use_napari :
        if ask_detection_confirmation(user_parameters.get('threshold')) : break
    else :
        break

result_df = pd.DataFrame()
cell_result_df = pd.DataFrame()
coloc_df = pd.DataFrame()

res, cell_res = launch_features_computation(
    acquisition_id=acquisition_id,
    image=image,
    nucleus_signal = nucleus_signal,
    dim=image.ndim,
    spots=spots,
    clusters=clusters,
    nucleus_label = nucleus_label,
    cell_label= cytoplasm_label,
    user_parameters=user_parameters,
    frame_results=frame_results,
    do_clustering=do_clustering
)

result_df = pd.concat([result_df, res])
cell_result_df = pd.concat([cell_result_df, cell_res])

while True :
    print(
        "result_df\n", result_df,
        "cell_result_df\n", cell_result_df,
        "coloc_df\n", coloc_df,
        )
    
    result_df, cell_result_df, coloc_df, acquisition_id, user_parameters = hub(
        acquisition_id, 
        result_df, 
        cell_result_df, 
        coloc_df, 
        segmentation_done=do_segmentation, 
        user_parameters=user_parameters, 
        cell_label=cytoplasm_label, 
        nucleus_label=nucleus_label
        )