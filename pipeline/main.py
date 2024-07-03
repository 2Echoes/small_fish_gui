#### New version for main.py
import pandas as pd
import PySimpleGUI as sg
from ..gui import hub_prompt
from .actions import add_detection, save_results, compute_colocalisation, delete_acquisitions
from ._preprocess import clean_unused_parameters_cache
from ..batch import batch_promp

#'Global' parameters
user_parameters = dict() # Very important object containg all choice from user that will influence the behavior of the main loop.
acquisition_id = -1
result_df = pd.DataFrame()
cell_result_df = pd.DataFrame()
coloc_df = pd.DataFrame()
segmentation_done = False
cytoplasm_label = None
nucleus_label = None

while True : #Break this loop to close small_fish
    result_df = result_df.reset_index(drop=True)
    cell_result_df = cell_result_df.reset_index(drop=True)
    coloc_df = coloc_df.reset_index(drop=True)
    try :
        event, values = hub_prompt(result_df, segmentation_done)

        if event == 'Add detection' :
            user_parameters = clean_unused_parameters_cache(user_parameters)

            new_result_df, new_cell_result_df, acquisition_id, user_parameters, segmentation_done, cytoplasm_label, nucleus_label =  add_detection(
                user_parameters=user_parameters,
                segmentation_done=segmentation_done,
                acquisition_id=acquisition_id,
                cytoplasm_label = cytoplasm_label,
                nucleus_label = nucleus_label,
                )
            result_df = pd.concat([result_df, new_result_df], axis=0)
            cell_result_df = pd.concat([cell_result_df, new_cell_result_df], axis=0)

        elif event == 'Save results' :
            save_results(
                result_df=result_df,
                cell_result_df=cell_result_df,
                coloc_df=coloc_df
            )
            
        elif event == 'Compute colocalisation' :
            result_tables = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.

            res_coloc= compute_colocalisation(
                result_tables,
                result_dataframe=result_df
            )

            coloc_df = pd.concat(
                [coloc_df,res_coloc],
                axis= 0)

        elif event == "Reset results" :
            result_df = pd.DataFrame()
            cell_result_df = pd.DataFrame()
            coloc_df = pd.DataFrame()
            acquisition_id = -1
            segmentation_done = False
            cytoplasm_label = None
            nucleus_label = None

        elif event == "Reset segmentation" :
            segmentation_done = False
            cytoplasm_label = None
            nucleus_label = None

        elif event == "Delete acquisitions" :
            selected_acquisitions = values.setdefault('result_table', []) #Contains the lines selected by the user on the sum-up array.
            result_df, cell_result_df, coloc_df = delete_acquisitions(selected_acquisitions, result_df, cell_result_df, coloc_df)

        elif event == "Batch detection" :
            result_df, cell_result_df, acquisition_id, user_parameters, segmentation_done, cytoplasm_label,nucleus_label = batch_promp(
                result_df,
                cell_result_df,
                acquisition_id=acquisition_id,
                preset=user_parameters,
            )
        
        else :
            break

    except Exception as error :
        sg.popup(str(error))
        raise error
quit()