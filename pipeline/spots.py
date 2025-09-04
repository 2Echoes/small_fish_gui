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
    
def load_spots(
        table_path : str
        ) -> pd.DataFrame :
    
    if table_path.endswith('.csv') :
        Spots = pd.read_csv(table_path, sep= ";")
    elif table_path.endswith('.xlsx') or table_path.endswith('.xls') :
        Spots = pd.read_excel(table_path)
    elif table_path.endswith('.feather') :
        Spots = pd.read_feather(table_path)
    else :
        raise ValueError("Table format not recognized. Please use .csv, .xlsx or .feather files.")
    
    if "coordinates" in Spots.columns :
        pass
    elif "y" in Spots.columns and "x" in Spots.columns :
        if "z" in Spots.columns :
            pass
        else :
            pass
    else :
        raise ValueError("Coordinates information not found in table. Please provide a 'coordinates' column with tuples (z,y,x) or (y,x) or 'y' and 'x' columns.")

    return Spots

def reconstruct_acquisition_data(
        Spots : pd.DataFrame,
        max_id : int,
        filename : str,
        ) :
    """
    Aim : creating a acquisition to add to result_dataframe from loaded spots for co-localization use  

    **Needed keys for colocalization**
        * acquisition_id
        * name
        * spots : np.ndarray[int] (nb_spots, nb_coordinates)
        * clusters : np.ndarray[int] (nb_cluster, nb_coordinate + 2)
        * spots_cluster_id : list[int]
        * voxel_size : tuple[int]
        * shape : tuple[int]
        * filename : str
    """
    
    spots = reconstruct_spots(Spots['coordinates'])
    has_clusters = not Spots['cluster_id'].isna().all()
    spot_number = len(spots)

    if has_clusters :

        clusters = np.empty(shape=(0,5), dtype=int) #useless for coloc only needded in columns to enable coloc on clusters
        spot_cluster_id = Spots['cluster_id'].to_numpy().astype(int).tolist()

        new_acquisition = pd.DataFrame({
            'acquisition_id' : [max_id + 1],
            'name' : ["loaded_spots_{}".format(max_id + 1)],
            'threshold' : [0],
            'spots' : [spots],
            'clusters' : [clusters],
            'spot_cluster_id' : [spot_cluster_id],
            'spot_number' : [spot_number],
            'filename' : [filename],
        })
    else :
        new_acquisition = pd.DataFrame({
            'acquisition_id' : [max_id + 1],
            'name' : ["loaded_spots_{}".format(max_id + 1)],
            'threshold' : [0],
            'spots' : [spots],
            'spot_number' : [spot_number],
            'filename' : [filename],
        })

    return new_acquisition

def reconstruct_spots(
        coordinates_serie : pd.Series
        ) :
    spots = coordinates_serie.str.replace('(','').str.replace(')','')
    spots = spots.str.split(',')
    spots = spots.apply(np.array)
    spots = np.array(spots.to_list()).astype(int)

    return spots


def reconstruct_cell_data(
        Spots : pd.DataFrame,
        max_id : int,
        ) :
    
    has_cluster = Spots['cluster_id'].isna().all()
    Spots['coordinates'] = reconstruct_spots(Spots['coordinates'])

    cell = Spots.groupby('cell_label')['coordinates'].apply(np.array).rename("rna_coords").reset_index(drop=False)
    cell['total_rna_number'] = cell['rna_coords'].apply(len)
    
    if has_cluster :
        cell = Spots[Spots['cluster_id'] !=-1].groupby('cell_label')['coordinates'].rename("clustered_spots_coords").apply(np.array).reset_index(drop=False)
        cell['clustered_spot_number'] = cell['clustered_spots_coords'].apply(len)

    cell['acquisition_id'] = max_id + 1
    cell['name'] = "loaded_spots_{}".format(max_id + 1)

    return cell