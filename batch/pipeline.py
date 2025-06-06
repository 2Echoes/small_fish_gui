"""
Submodule keeping necessary calls from main pipeline for batch processing.
"""

import os
import pandas as pd
import FreeSimpleGUI as sg

from ..hints import pipeline_parameters

from .input import open_image
from ..interface import write_results
from ..pipeline import reorder_shape, reorder_image_stack, prepare_image_detection
from ..pipeline import cell_segmentation, launch_detection, launch_features_computation
from ..pipeline import launch_spots_extraction
from ..pipeline import get_nucleus_signal
from ..pipeline import _cast_segmentation_parameters, convert_parameters_types
from ..pipeline import plot_segmentation, output_spot_tiffvisual
from ..utils import get_datetime

def window_print(window: sg.Window, *args) :
    print(*args)
    window.refresh()

def batch_pipeline(
        batch_window : sg.Window,
        batch_progress_bar : sg.ProgressBar,
        progress_count : sg.Text,
        parameters : pipeline_parameters,
        filenames_list : list,
        do_segmentation : bool,
        map_ : dict,
        results_df : pd.DataFrame,
        cell_results_df : pd.DataFrame,
        is_3D,
        last_acquisition_id=0,
) :
    
    #Extracting parameters
    input_path = parameters['Batch_folder']
    output_path = parameters['output_folder']
    batch_name = parameters['batch_name']
    time = '_' + get_datetime()

    #Preparing folder
    window_print(batch_window,"Creating folders for output...")
    main_dir = output_path + "/" + batch_name + time + "/"
    os.makedirs(main_dir + "results/", exist_ok=True)
    if parameters['save segmentation'] : os.makedirs(main_dir + "segmentation/", exist_ok=True)
    if parameters['save detection'] : os.makedirs(main_dir + "detection/", exist_ok=True)
    if parameters['extract spots'] : os.makedirs(main_dir + "results/spots_extraction", exist_ok=True)
    first_save = True # init for excel append
    append_to_line = 1 #Start at line one
    cell_append_to_line = 1 #Start at line one

    #Setting spot detection dimension
    parameters['dim'] = 3 if is_3D else 2

    #Pipeline loop
    window_print(batch_window,"Launching batch analysis...")
    batch_progress_bar.update(max=len(filenames_list))
    filenames_list.sort()
    

    error_log = open(main_dir + "error_log", mode='w')
    error_count = 0

    #These columns usually kept for coloc analysis will be dropped for memory gain in batch mode
    COLUMNS_TO_DROP = ['image', 'spots', 'clusters', 'rna_coords', 'cluster_coords',"rna_coords", "cluster_coords", "free_spots_coords", "clustered_spots_coords"]
    for col in COLUMNS_TO_DROP :
                if col in cell_results_df : cell_results_df.drop(columns=col)
                if col in results_df : results_df.drop(columns=col)

    for acquisition_id, file in enumerate(filenames_list) :
        try :

            #GUI
            window_print(batch_window,"\nNext file : {0}".format(file))
            batch_progress_bar.update(current_count= acquisition_id, max= len(filenames_list))
            progress_count.update(value=str(acquisition_id))
            batch_window = batch_window.refresh()

            #0. Open image
            image = open_image(input_path + '/' + file)
            parameters['image'] = image
            parameters['filename'] = file
            for key_to_clean in [0,2] : 
                if key_to_clean in parameters : del parameters[key_to_clean]

            #1. Re-order shape
            shape = image.shape
            parameters['shape'] = shape
            parameters['reordered_shape'] = reorder_shape(shape, map_=map_)

            #2. Segmentation (opt)
            if do_segmentation :
                window_print(batch_window,"Segmenting cells...")
                im_seg = reorder_image_stack(map_, image)
                parameters = _cast_segmentation_parameters(parameters)
                cytoplasm_label, nucleus_label = cell_segmentation(
                    im_seg,
                    cyto_model_name= parameters['cyto_model_name'],
                    cyto_diameter= parameters['cytoplasm diameter'],
                    nucleus_model_name= parameters['nucleus_model_name'],
                    nucleus_diameter= parameters['nucleus diameter'],
                    channels=[parameters['cytoplasm channel'], parameters['nucleus channel']],
                    do_only_nuc=parameters['Segment only nuclei']
                    )

                parameters['segmentation_done'] = True

                if cytoplasm_label.max() == 0 : #No cell segmented
                    window_print(batch_window,"No cell was segmented, computing next image.")
                    continue
                else : 
                    window_print(batch_window, "{0} cells segmented.".format(cytoplasm_label.max()))

                    if parameters['save segmentation'] :
                        plot_segmentation(
                            cyto_image=im_seg[parameters['cytoplasm channel']],
                            cyto_label= cytoplasm_label,
                            nuc_image= im_seg[parameters['nucleus channel']],
                            nuc_label=nucleus_label,
                            path= main_dir + "segmentation/" + file,
                            do_only_nuc= parameters['Segment only nuclei'],
                        )

            else :
                cytoplasm_label, nucleus_label = None,None
                parameters['segmentation_done'] = False

            #3. Detection, deconvolution, clusterisation
            window_print(batch_window,"Detecting spots...")
            parameters = convert_parameters_types(parameters)
            image, other_image = prepare_image_detection(map_, parameters) 
            nucleus_signal = get_nucleus_signal(image, other_image, parameters)
            try : # Catch error raised if user enter a spot size too small compare to voxel size
                parameters, frame_result, spots, clusters, spot_cluster_id = launch_detection(
                    image,
                    other_image,
                    parameters,
                    cell_label=cytoplasm_label,
                    nucleus_label=nucleus_label,
                    hide_loading=True,
                )

            except ValueError as error :
                if "The array should have an upper bound of 1" in str(error) :
                    window_print(batch_window,"Spot size too small for current voxel size.")
                    continue
                else :
                    raise(error)

            if parameters['save detection'] :
                if parameters['do_cluster_computation'] : 
                    if len(clusters) > 0 :
                        spots_list = [spots, clusters[:,:-2]]
                    else : spots_list = [spots]
                else : spots_list = [spots]
                output_spot_tiffvisual(
                    image,
                    spots_list= spots_list,
                    dot_size=2,
                    path_output= main_dir + "detection/" + file + "_spot_detection.tiff"
                )

            #4. Spots extraction
            window_print(batch_window,"Extracting spots : ")
            if parameters['extract spots'] :

                #Setting parameter for call to lauch spot extraction
                #Only spots have one file per image to avoir memory overload
                parameters['do_spots_excel'] = parameters['xlsx']
                parameters['do_spots_csv'] = parameters['csv']
                parameters['do_spots_feather'] = parameters['feather']
                parameters['spots_filename'] = "spots_extractions_{0}".format(file)
                parameters['spots_extraction_folder'] = main_dir + "results/spots_extraction/"

                launch_spots_extraction(
                        acquisition_id=acquisition_id + last_acquisition_id,
                        user_parameters=parameters,
                        image=image,
                        spots=spots,
                        cluster_id=spot_cluster_id,
                        nucleus_label= nucleus_label,
                        cell_label= cytoplasm_label,
                    )

            #5. Features computation
            window_print(batch_window,"computing features...")
            new_results_df, new_cell_results_df = launch_features_computation(
            acquisition_id=acquisition_id + last_acquisition_id,
            image=image,
            nucleus_signal = nucleus_signal,
            spots=spots,
            clusters=clusters,
            spots_cluster_id=spot_cluster_id,
            nucleus_label = nucleus_label,
            cell_label= cytoplasm_label,
            user_parameters=parameters,
            frame_results=frame_result,
            )

            for col in COLUMNS_TO_DROP :
                if col in new_cell_results_df : new_cell_results_df.drop(columns=col)
                if col in new_results_df : new_results_df.drop(columns=col)

            results_df = pd.concat([
                results_df.reset_index(drop=True), new_results_df.reset_index(drop=True)
            ], axis=0)

            cell_results_df = pd.concat([
                cell_results_df.reset_index(drop=True), new_cell_results_df.reset_index(drop=True)
            ], axis=0)


            #6. Saving results
            if parameters['xlsx'] :
                if first_save : xlsx_header = True
                else : xlsx_header = False

            window_print(batch_window,"saving image_results...")
            #1 file per batch + 1 file per batch if segmentation
            acquisition_success = write_results(
                results_df, 
                path= main_dir + "results/", 
                filename=batch_name, 
                do_excel= parameters["xlsx"], 
                do_feather= parameters["feather"], 
                do_csv= parameters["csv"],
                overwrite=True,
                batch_mode=True,
                header=first_save,
                xlsx_start_line=append_to_line
                )
            append_to_line += len(results_df)
            results_df = results_df.drop(results_df.index)

            if do_segmentation :
                cell_success = write_results(
                    cell_results_df, 
                    path= main_dir + "results/", 
                    filename=batch_name + '_cell_result', 
                    do_excel= parameters["xlsx"], 
                    do_feather= parameters["feather"], 
                    do_csv= parameters["csv"],
                    overwrite=True,
                    batch_mode=True,
                    header=first_save,
                    xlsx_start_line=cell_append_to_line
                    )
                cell_append_to_line += len(cell_results_df)
                cell_results_df = cell_results_df.drop(cell_results_df.index)
            first_save = False
            window_print(batch_window,"Sucessfully saved.")


        except Exception as error :
            
            error_count +=1
            print("Exception raised for acquisition, writting error in error log.")

            log = [
                f"Error raised during acquisition {acquisition_id}.\n"
            ]
            log.append(str(error) + '\n')

            error_log.writelines(log)

            print("Ignoring current acquisition and proceeding to next one.")
            continue

    
        
    batch_progress_bar.update(current_count= acquisition_id+1, max= len(filenames_list))
    progress_count.update(value=str(acquisition_id+1))
    batch_window = batch_window.refresh()

    if error_count > 0 :
        sg.popup(f"Batch processing finished but {error_count} acquisitions were skipped during quantification.\nFor more informations check error_log in result folder.")

    return results_df, cell_results_df, acquisition_id
