"""
Sub-module to handle individual spot extraction.

"""

import numpy as np
import pandas as pd
from ..interface.inoutput import write_results

def launch_spots_extraction(
        acquisition_id,
        user_parameters,
        image,
        spots,
        cluster_id,
        nucleus_label,
        cell_label,
) :
    Spots = compute_Spots(
        acquisition_id=acquisition_id,
        image=image,
        spots=spots,
        cluster_id= cluster_id,
        nucleus_label=nucleus_label,
        cell_label=cell_label,
    )

    did_output = write_results(
        Spots,
        path= user_parameters['spots_extraction_folder'],
        filename= user_parameters['spots_filename'],
        do_excel=user_parameters['do_spots_excel'],
        do_csv=user_parameters['do_spots_csv'],
        do_feather=user_parameters['do_spots_feather'],
        )
    
    if did_output : print("Individual spots extracted at {0}".format(user_parameters['spots_extraction_folder']))

def compute_Spots(
        acquisition_id : int,
        image : np.ndarray,
        spots : np.ndarray,
        cluster_id : np.ndarray,
        nucleus_label = None,
        cell_label = None,
) :

    if len(spots) == 0 :
        return pd.DataFrame()

    if type(cluster_id) == type(None) : #When user doesn't select cluster
        cluster_id = [np.NaN]*len(spots)

    index = list(zip(*spots))
    index = tuple(index)
    spot_intensities_list = list(image[index])
    if type(nucleus_label) != type(None) :
        if nucleus_label.ndim == 2 :
            in_nuc_list = list(nucleus_label.astype(bool)[index[-2:]]) #Only plane coordinates
        else :
            in_nuc_list = list(nucleus_label.astype(bool)[index])
    else :
        in_nuc_list = np.NaN
    if type(cell_label) != type(None) :
        cell_label_list = list(cell_label[index[-2:]]) #Only plane coordinates
    else :
        cell_label_list = np.NaN
    id_list = np.arange(len(spots))

    coord_list = list(zip(*index))

    Spots = pd.DataFrame({
        'acquisition_id' : [acquisition_id] * len(spots),
        'spot_id' : id_list,
        'intensity' : spot_intensities_list,
        'cell_label' : cell_label_list,
        'in_nucleus' : in_nuc_list,
        'coordinates' : coord_list,
        'cluster_id' : cluster_id,
    })

    return Spots
    

