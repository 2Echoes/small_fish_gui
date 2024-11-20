"""
This script is called when software starts; it is the main loop.
"""

import pandas as pd
import PySimpleGUI as sg
from ..gui import hub_prompt
from .actions import add_detection, save_results, compute_colocalisation, delete_acquisitions, rename_acquisitions, save_segmentation, load_segmentation, segment_cells
from ._preprocess import clean_unused_parameters_cache
from ..batch import batch_promp
from .hints import pipeline_parameters

#'Global' parameters
user_parameters = pipeline_parameters({'segmentation_done' : False}) #TypedDict
acquisition_id = -1
result_df = pd.DataFrame()
cell_result_df = pd.DataFrame()
global_coloc_df = pd.DataFrame()
cell_coloc_df = pd.DataFrame()
cytoplasm_label = None
nucleus_label = None

#Use for dev purpose
MAKE_NEW_SAVE = False
PATH = "/home/floricslimani/Documents/small_fish_workshop/save"
LOAD_SAVE = False

while True : #Break this loop to close small_fish

    if LOAD_SAVE :
        result_df = pd.read_csv(PATH + "/result.csv", sep='|')
        cell_result_df = pd.read_csv(PATH + "/cell_result_df.csv", sep='|')
        global_coloc_df = pd.read_csv(PATH + "/global_coloc_df.csv", sep='|')
        cell_coloc_df = pd.read_csv(PATH + "/cell_coloc_df.csv", sep='|')


    else :
        result_df = result_df.reset_index(drop=True)
        cell_result_df = cell_result_df.reset_index(drop=True)
        global_coloc_df = global_coloc_df.reset_index(drop=True)
        cell_coloc_df = cell_coloc_df.reset_index(drop=True)
    try :
        event, values = hub_prompt(result_df, user_parameters['segmentation_done'])

        if event == 'Add detection' :
            user_parameters = clean_unused_parameters_cache(user_parameters)

            new_result_df, new_cell_result_df, acquisition_id, user_parameters =  add_detection(
                user_parameters=user_parameters,
                acquisition_id=acquisition_id,
                cytoplasm_label = cytoplasm_label,
                nucleus_label = nucleus_label,
                )
            result_df = pd.concat([result_df, new_result_df], axis=0)
            cell_result_df = pd.concat([cell_result_df, new_cell_result_df], axis=0)

        elif event == 'Segment cells' :
            nucleus_label, cytoplasm_label, user_parameters = segment_cells(
                user_parameters=user_parameters,
                nucleus_label=nucleus_label,
                cytoplasm_label = cytoplasm_label,
            )

        elif event == 'Save results' :
            save_results(
                result_df=result_df,
                cell_result_df=cell_result_df,
                global_coloc_df=global_coloc_df,
                cell_coloc_df = cell_coloc_df,
            )
        
        elif event == 'Save segmentation' :
            save_segmentation(
                nucleus_label=nucleus_label,
                cytoplasm_label=cytoplasm_label,
            )

        elif event == 'Load segmentation' :
            nucleus_label, cytoplasm_label, user_parameters['segmentation_done'] = load_segmentation(nucleus_label, cytoplasm_label, user_parameters['segmentation_done'])
        
        elif event == 'Compute colocalisation' :
            result_tables = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.

            global_coloc_df, cell_coloc_df = compute_colocalisation(
                result_tables,
                result_dataframe=result_df,
                cell_result_dataframe=cell_result_df,
                global_coloc_df=global_coloc_df,
                cell_coloc_df=cell_coloc_df,
            )

        elif event == "Reset results" :
            result_df = pd.DataFrame()
            cell_result_df = pd.DataFrame()
            global_coloc_df = pd.DataFrame()
            cell_coloc_df = pd.DataFrame()
            acquisition_id = -1
            user_parameters['segmentation_done'] = False
            cytoplasm_label = None
            nucleus_label = None

        elif event == "Reset segmentation" :
            user_parameters['segmentation_done'] = False
            cytoplasm_label = None
            nucleus_label = None

        elif event == "Delete acquisitions" :
            selected_acquisitions = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.
            result_df, cell_result_df, global_coloc_df, cell_coloc_df = delete_acquisitions(selected_acquisitions, result_df, cell_result_df, global_coloc_df, cell_coloc_df)

        elif event == "Batch detection" :
            result_df, cell_result_df, acquisition_id, user_parameters, user_parameters['segmentation_done'], cytoplasm_label,nucleus_label = batch_promp(
                result_df,
                cell_result_df,
                acquisition_id=acquisition_id,
                preset=user_parameters,
            )
        
        elif event == "Rename acquisition" :
            selected_acquisitions = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.
            result_df, cell_result_df, global_coloc_df, cell_coloc_df = rename_acquisitions(selected_acquisitions, result_df, cell_result_df, global_coloc_df, cell_coloc_df)

        else :
            break
        
        if MAKE_NEW_SAVE :
            result_df.reset_index(drop=True).to_csv(PATH + "/result.csv", sep='|')
            cell_result_df.reset_index(drop=True).to_csv(PATH + "/cell_result_df.csv", sep='|')
            cell_coloc_df.reset_index(drop=True).to_csv(PATH + "/cell_coloc_df.csv", sep='|')
            global_coloc_df.reset_index(drop=True).to_csv(PATH + "/global_coloc_df.csv", sep='|')

    except Exception as error :
        sg.popup(str(error))
        raise error
quit()