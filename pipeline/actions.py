from ..gui.prompts import output_image_prompt, ask_detection_confirmation, ask_cancel_detection
from ..interface.output import write_results
from ._preprocess import map_channels, prepare_image_detection, reorder_shape, reorder_image_stack
from .detection import ask_input_parameters, initiate_detection, launch_detection, launch_features_computation, get_nucleus_signal
from ._segmentation import launch_segmentation
from ._colocalisation import initiate_colocalisation, launch_colocalisation

import pandas as pd
import PySimpleGUI as sg

def add_detection(user_parameters, segmentation_done, acquisition_id, cytoplasm_label, nucleus_label) :
    """
    #TODO : list all keys added to user_parameters when returned
    """

    new_results_df = pd.DataFrame()
    new_cell_results_df = pd.DataFrame()

    #Ask for image parameters
    new_parameters = ask_input_parameters(ask_for_segmentation= not segmentation_done) #The image is open and stored inside user_parameters
    if type(new_parameters) == type(None) : #if user clicks 'Cancel'
        return new_results_df, new_cell_results_df, acquisition_id, user_parameters, segmentation_done,cytoplasm_label, nucleus_label
    else :
        user_parameters.update(new_parameters)

    map = map_channels(user_parameters)
    user_parameters['reordered_shape'] = reorder_shape(user_parameters['shape'], map)


    #Segmentation
    if user_parameters['Segmentation'] and not segmentation_done:
        im_seg = reorder_image_stack(map, user_parameters)
        cytoplasm_label, nucleus_label, user_parameters = launch_segmentation(im_seg, user_parameters=user_parameters)

    else :
        cytoplasm_label, nucleus_label = None,None

    if type(cytoplasm_label) == type(None) or type(nucleus_label) == type(None) :
        user_parameters['Segmentation'] = False
        segmentation_done = False

    else : segmentation_done = True

    #Detection
    while True : # This loop allow user to try detection with different thresholds or parameters before launching features computation
        detection_parameters = initiate_detection(
            user_parameters,
            segmentation_done,
            map= map,
            shape = user_parameters['image'].shape
            )

        if type(detection_parameters) != type(None) :
            user_parameters.update(detection_parameters) 
        else : #If user clicks cancel
            cancel = ask_cancel_detection()
            if cancel : 
                return new_results_df, new_cell_results_df, acquisition_id, user_parameters, segmentation_done, cytoplasm_label, nucleus_label
            else : continue

        acquisition_id += 1
        image, other_image = prepare_image_detection(map, user_parameters) 
        nucleus_signal = get_nucleus_signal(image, other_image, user_parameters)
        
        try : # Catch error raised if user enter a spot size too small compare to voxel size
            user_parameters, frame_result, spots, clusters = launch_detection(
                image,
                other_image,
                user_parameters,
                cell_label=cytoplasm_label,
                nucleus_label=nucleus_label
            )

        except ValueError as error :
            if "The array should have an upper bound of 1" in str(error) :
                sg.popup("Spot size too small for current voxel size.")
                continue
            else :
                raise(error)


        if user_parameters['Napari correction'] :
            if ask_detection_confirmation(user_parameters.get('threshold')) : break
        else :
            break
        
    #Features computation
    new_results_df, new_cell_results_df = launch_features_computation(
    acquisition_id=acquisition_id,
    image=image,
    nucleus_signal = nucleus_signal,
    spots=spots,
    clusters=clusters,
    nucleus_label = nucleus_label,
    cell_label= cytoplasm_label,
    user_parameters=user_parameters,
    frame_results=frame_result,
    )
    return new_results_df, new_cell_results_df, acquisition_id, user_parameters, segmentation_done, cytoplasm_label, nucleus_label

def save_results(result_df, cell_result_df, coloc_df) :
    if len(result_df) != 0 :
        dic = output_image_prompt(filename=result_df.iloc[0].at['filename'])

        if isinstance(dic, dict) :
            path = dic['folder']
            filename = dic['filename']
            do_excel = dic['Excel']
            do_feather = dic['Feather']
            sucess1 = write_results(result_df, path= path, filename=filename, do_excel= do_excel, do_feather= do_feather)
            sucess2 = write_results(cell_result_df, path= path, filename=filename + '_cell_result', do_excel= do_excel, do_feather= do_feather)
            sucess3 = write_results(coloc_df, path= path, filename=filename + '_coloc_result', do_excel= do_excel, do_feather= do_feather)
            if sucess1 and sucess2 and sucess3 : sg.popup("Sucessfully saved at {0}.".format(path))

    else :
        dic = None
        sg.popup('No results to save.') 

def compute_colocalisation(result_tables, result_dataframe) :
    colocalisation_distance = initiate_colocalisation(result_tables)

    if colocalisation_distance == False :
        res_coloc = pd.DataFrame() # popup handled in initiate_colocalisation
    else :
        res_coloc = launch_colocalisation(result_tables, result_dataframe=result_dataframe, colocalisation_distance=colocalisation_distance)

    return res_coloc