"""
This submodule groups all the possible actions of the user in the main windows. It is the start of each action the user can do.
"""

from ..gui.prompts import output_image_prompt, prompt_save_segmentation, prompt_load_segmentation
from ..gui.prompts import ask_detection_confirmation, ask_cancel_detection, ask_confirmation
from ..gui.prompts import rename_prompt

from ..interface.inoutput import write_results
from ..interface.inoutput import input_segmentation, output_segmentation

from ._preprocess import map_channels
from ._preprocess import prepare_image_detection
from ._preprocess import reorder_shape, reorder_image_stack
from ._preprocess import ask_input_parameters

from .detection import initiate_detection, launch_detection, launch_features_computation
from .detection import get_nucleus_signal
from .spots import launch_spots_extraction

from .segmentation import launch_segmentation
from ._colocalisation import initiate_colocalisation, launch_colocalisation

from ..hints import pipeline_parameters

import os
import pandas as pd
import PySimpleGUI as sg
import numpy as np

def segment_cells(user_parameters : pipeline_parameters, nucleus_label, cytoplasm_label) :
    if user_parameters['segmentation_done'] :
        if ask_confirmation("A segmentation is in small fish memory, do you want to erase it ? (you will not be able to undo)") :
            user_parameters['segmentation_done'] = False
            nucleus_label = None
            cytoplasm_label = None
        else : 
            return nucleus_label, cytoplasm_label, user_parameters

    nucleus_label, cytoplasm_label, user_parameters = launch_segmentation(
        user_parameters,
        nucleus_label=nucleus_label,
        cytoplasm_label=cytoplasm_label,
    )

    if type(cytoplasm_label) != type(None) and type(nucleus_label) != type(None) :
        user_parameters['segmentation_done'] = True

    return nucleus_label, cytoplasm_label, user_parameters

def add_detection(user_parameters : pipeline_parameters, acquisition_id, cytoplasm_label, nucleus_label) :
    """
    #TODO : Separate segmentation from detection in pipeline.
    """

    new_results_df = pd.DataFrame()
    new_cell_results_df = pd.DataFrame()

    #Ask for image parameters
    new_parameters = ask_input_parameters(ask_for_segmentation= False) #The image is open and stored inside user_parameters
    if type(new_parameters) == type(None) : #if user clicks 'Cancel'
        return new_results_df, new_cell_results_df, acquisition_id, user_parameters
    else :
        user_parameters.update(new_parameters)

    map_ = map_channels(user_parameters)
    if type(map_) == type(None) : #User clicks Cancel 
        return new_results_df, new_cell_results_df, acquisition_id, user_parameters
    user_parameters['reordered_shape'] = reorder_shape(user_parameters['shape'], map_)

    #Detection
    while True : # This loop allow user to try detection with different thresholds or parameters before launching features computation
        detection_parameters = initiate_detection(
            user_parameters,
            map_= map_,
            shape = user_parameters['image'].shape
            )

        if type(detection_parameters) != type(None) :
            user_parameters.update(detection_parameters) 
        else : #If user clicks cancel
            cancel = ask_cancel_detection()
            if cancel : 
                return new_results_df, new_cell_results_df, acquisition_id, user_parameters
            else : continue

        acquisition_id += 1
        image, other_image = prepare_image_detection(map_, user_parameters) 
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


        if user_parameters['show_napari_corrector'] :
            if ask_detection_confirmation(user_parameters.get('threshold')) : break
        else :
            break

    if user_parameters['spots_extraction_folder'] != '' and type(user_parameters['spots_extraction_folder']) != type(None) :
        if user_parameters['spots_filename'] != '' and type(user_parameters['spots_filename']) != type(None) :
            if any((user_parameters['do_spots_excel'], user_parameters['do_spots_csv'], user_parameters['do_spots_feather'])) :
                launch_spots_extraction(
                    acquisition_id=acquisition_id,
                    user_parameters=user_parameters,
                    image=image,
                    spots=spots,
                    nucleus_label= nucleus_label,
                    cell_label= cytoplasm_label,
                )

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
    return new_results_df, new_cell_results_df, acquisition_id, user_parameters

def save_segmentation(nucleus_label : np.ndarray, cytoplasm_label: np.ndarray) :
    if type(nucleus_label) == type(None) or type(cytoplasm_label) == type(None) :
        sg.popup("No segmentation to save.")
    
    else :

        while True :
            answer = prompt_save_segmentation()
            if type(answer) == type(None) : 
                return False #User clicks cancel

            path = answer['folder'] + '/' + answer['filename']
            is_npy, is_npz, is_npz_compressed = answer['ext'], answer['ext0'], answer['ext1']

            if is_npy + is_npz + is_npz_compressed == 1 : 
                if is_npy : 
                    extension = 'npy'
                    if os.path.isfile(path + '_nucleus_segmentation.npy') or os.path.isfile(path + '_cytoplasm_segmentation.npy') : 
                        if ask_confirmation("File exists. Replace ?") :
                            break
                        else :
                            pass
                    else :
                        break

                elif is_npz :
                    extension = 'npz_uncompressed'
                    if os.path.isfile(path + '_nucleus_segmentation.npz') or os.path.isfile(path + '_cytoplasm_segmentation.npz') : 
                        if ask_confirmation("File exists. Replace ?") :
                            break
                        else :
                            pass
                    else :
                        break

                elif is_npz_compressed : 
                    extension = 'npz_compressed'
                    if os.path.isfile(path + '_nucleus_segmentation.npz') or os.path.isfile(path + '_cytoplasm_segmentation.npz') : 
                        if ask_confirmation("File exists. Replace ?") :
                            break
                        else :
                            pass
                    else :
                        break

            else :
                sg.popup("Please select an extension.")

        saved = output_segmentation(
            path,
            extension,
            nucleus_label,
            cytoplasm_label,
            )
        if saved : sg.popup("Segmentation was saved at {0}.".format(path))
        else : sg.popup("No segmentation was saved..")
        return True
    
def load_segmentation(nucleus_label, cytoplasm_label, segmentation_done) :

    if segmentation_done :
        if ask_confirmation("Segmentation already in memory. Replace ?") :
            pass
        else :
            return nucleus_label, cytoplasm_label, segmentation_done

    answer = prompt_load_segmentation()
    if type(answer) == type(None) : #user clicks cancel
        return nucleus_label, cytoplasm_label, segmentation_done
    nucleus_label, cytoplasm_label = input_segmentation(
        answer['nucleus'],
        answer['cytoplasm'],
    )

    if type(nucleus_label) != type(None) and type(nucleus_label) != np.ndarray :
        nucleus_label = nucleus_label['arr_0']

    if type(cytoplasm_label) != type(None) and type(cytoplasm_label) != np.ndarray :
        cytoplasm_label = cytoplasm_label['arr_0']

    segmentation_done = (type(nucleus_label) != type(None) and type(cytoplasm_label) != type(None))

    if segmentation_done : assert type(nucleus_label) == np.ndarray and type(cytoplasm_label) == np.ndarray

    return nucleus_label, cytoplasm_label, segmentation_done

def save_results(result_df, cell_result_df, global_coloc_df, cell_coloc_df) :
    if len(result_df) != 0 :
        dic = output_image_prompt(filename=result_df.iloc[0].at['filename'])

        if isinstance(dic, dict) :
            path = dic['folder']
            filename = dic['filename']
            do_excel = dic['Excel']
            do_feather = dic['Feather']
            do_csv = dic['csv']

            if 'rna_coords' in cell_result_df.columns : cell_result_df = cell_result_df.drop(columns='rna_coords')

            sucess1 = write_results(result_df, path= path, filename=filename, do_excel= do_excel, do_feather= do_feather, do_csv=do_csv)
            sucess2 = write_results(cell_result_df, path= path, filename=filename + '_cell_result', do_excel= do_excel, do_feather= do_feather, do_csv=do_csv)
            sucess3 = write_results(global_coloc_df, path= path, filename=filename + 'global_coloc_result', do_excel= do_excel, do_feather= do_feather, do_csv=do_csv)
            sucess4 = write_results(cell_coloc_df, path= path, filename=filename + 'cell2cell_coloc_result', do_excel= do_excel, do_feather= do_feather, do_csv=do_csv, reset_index=False)
            if all([sucess1,sucess2, sucess3, sucess4,]) : sg.popup("Sucessfully saved at {0}.".format(path))

    else :
        dic = None
        sg.popup('No results to save.') 

def compute_colocalisation(result_tables, result_dataframe, cell_result_dataframe, global_coloc_df, cell_coloc_df) :
    colocalisation_distance = initiate_colocalisation(result_tables)

    if colocalisation_distance == False :
        pass
    else :
        global_coloc_df, cell_coloc_df = launch_colocalisation(
            result_tables, 
            result_dataframe=result_dataframe, 
            cell_result_dataframe=cell_result_dataframe,
            colocalisation_distance=colocalisation_distance,
            global_coloc_df=global_coloc_df,
            cell_coloc_df=cell_coloc_df,
            )

    return global_coloc_df, cell_coloc_df

def delete_acquisitions(selected_acquisitions : pd.DataFrame, 
                        result_df : pd.DataFrame, 
                        cell_result_df : pd.DataFrame, 
                        global_coloc_df : pd.DataFrame,
                        cell_coloc_df : pd.DataFrame,
                        ) :
    
    if len(result_df) == 0 :
        sg.popup("No acquisition to delete.")
        return result_df, cell_result_df, global_coloc_df

    if len(selected_acquisitions) == 0 :
        sg.popup("Please select the acquisitions you would like to delete.")
    else :
        acquisition_ids = list(result_df.iloc[list(selected_acquisitions)]['acquisition_id'])
        result_drop_idx = result_df[result_df['acquisition_id'].isin(acquisition_ids)].index
        print("{0} acquisitions deleted.".format(len(result_drop_idx)))
        
        if len(cell_result_df) > 0 :
            cell_result_df_drop_idx = cell_result_df[cell_result_df['acquisition_id'].isin(acquisition_ids)].index
            print("{0} cells deleted.".format(len(cell_result_df_drop_idx)))
            cell_result_df = cell_result_df.drop(cell_result_df_drop_idx, axis=0)
        
        if len(global_coloc_df) > 0 :
            coloc_df_drop_idx = global_coloc_df[(global_coloc_df["acquisition_id_1"].isin(acquisition_ids)) | (global_coloc_df['acquisition_id_2'].isin(acquisition_ids))].index
            print("{0} coloc measurement deleted.".format(len(coloc_df_drop_idx)))
            global_coloc_df = global_coloc_df.drop(coloc_df_drop_idx, axis=0)
        
        if len(cell_coloc_df) > 0 :
            for acquisition_id in acquisition_ids :
                cell_coloc_df = cell_coloc_df.drop(acquisition_id, axis=1, level=2) #Delete spot number and foci number
                coloc_columns = cell_coloc_df.columns.get_level_values(1)
                coloc_columns = coloc_columns[coloc_columns.str.contains(str(acquisition_id))]
                cell_coloc_df = cell_coloc_df.drop(labels=coloc_columns, axis=1, level=1)

        result_df = result_df.drop(result_drop_idx, axis=0)

    return result_df, cell_result_df, global_coloc_df, cell_coloc_df

def rename_acquisitions(
        selected_acquisitions : pd.DataFrame, 
        result_df : pd.DataFrame, 
        cell_result_df : pd.DataFrame, 
        global_coloc_df : pd.DataFrame,
        cell_coloc_df : pd.DataFrame,
        ) :
    
    if len(result_df) == 0 :
        sg.popup("No acquisition to rename.")
        return result_df, cell_result_df, global_coloc_df, cell_coloc_df
    
    if len(selected_acquisitions) == 0 :
        sg.popup("Please select the acquisitions you would like to rename.")

    else :
        name = rename_prompt()
        if not name : return result_df, cell_result_df, global_coloc_df, cell_coloc_df #User didn't put a name or canceled
        name : str = name.replace(' ','_')
        acquisition_ids = list(result_df.iloc[list(selected_acquisitions)]['acquisition_id'])
        old_names = list(result_df.loc[result_df['acquisition_id'].isin(acquisition_ids)]['name'])
        old_names.sort(key=len) #We order this list by elmt length
        old_names.reverse() #From longer to smaller

        result_df.loc[result_df['acquisition_id'].isin(acquisition_ids),['name']] = name
        if len(cell_result_df) > 0 : cell_result_df.loc[cell_result_df['acquisition_id'].isin(acquisition_ids),['name']] = name
        if len(global_coloc_df) > 0 : 
            global_coloc_df.loc[global_coloc_df['acquisition_id_1'].isin(acquisition_ids), ['name1']] = name
            global_coloc_df.loc[global_coloc_df['acquisition_id_2'].isin(acquisition_ids), ['name2']] = name
        if len(cell_coloc_df) > 0 :
            target_columns = cell_coloc_df.columns.get_level_values(1)
            for old_name in old_names : #Note list was ordered by elmt len (decs) to avoid conflict when one name is contained by another one. if the shorter is processed first then the longer will not be able to be properly renamed.
                target_columns = target_columns.str.replace(old_name, name)
            
            new_columns = zip(
                cell_coloc_df.columns.get_level_values(0),
                target_columns,
                cell_coloc_df.columns.get_level_values(2),
            )

            cell_coloc_df.columns = pd.MultiIndex.from_tuples(new_columns)

    return result_df, cell_result_df, global_coloc_df, cell_coloc_df