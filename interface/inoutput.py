import os
import pandas as pd
import numpy as np
from bigfish.stack import check_parameter
from typing import Literal

MAX_LEN_EXCEL = 1048576

def _cast_spot_to_tuple(spot) :
    return tuple([coord for coord in spot])

def _cast_spots_to_tuple(spots) :
    return tuple(list(map(_cast_spot_to_tuple, spots)))

def write_results(dataframe: pd.DataFrame, path:str, filename:str, do_excel= True, do_feather= False, do_csv=False, overwrite=False, reset_index=True) :
    check_parameter(dataframe= pd.DataFrame, path= str, filename = str, do_excel = bool, do_feather = bool)

    if len(dataframe) == 0 : return True
    if not do_excel and not do_feather and not do_csv : 
        return False

    if not path.endswith('/') : path +='/'
    assert os.path.isdir(path)

    #Casting cols name to str for feather format
    index_dim = dataframe.columns.nlevels
    if index_dim == 1 :
        dataframe.columns = dataframe.columns.astype(str)
    else :
        casted_cols = [dataframe.columns.get_level_values(level).astype(str) for level in range(index_dim)]
        casted_cols = zip(*casted_cols)
        dataframe.columns = pd.MultiIndex.from_tuples(casted_cols)

    new_filename = filename
    i= 1

    if not overwrite :
        while new_filename + '.xlsx' in os.listdir(path) or new_filename + '.parquet' in os.listdir(path) or new_filename + '.csv' in os.listdir(path) :
            new_filename = filename + '_{0}'.format(i)
            i+=1

    COLUMNS_TO_DROP = ['image', 'spots', 'clusters', 'rna_coords', 'cluster_coords']
    for col in COLUMNS_TO_DROP :
        if col in dataframe.columns : dataframe = dataframe.drop(columns=col)

    if reset_index : dataframe = dataframe.reset_index(drop=True)

    if do_csv : dataframe.to_csv(path + new_filename + '.csv', sep=";")
    if do_excel : 
        if len(dataframe) < MAX_LEN_EXCEL :
            dataframe.to_excel(path + new_filename + '.xlsx')
        else : 
            print("Error : Table too big to be saved in excel format.")
            return False
    
    if do_feather : 
        dataframe.to_parquet(path + new_filename + '.parquet')

    return True

def input_segmentation(
        nucleus_path : str,
        cytoplasm_path : str,
) :
    nucleus_label = np.load(nucleus_path)

    if cytoplasm_path != '' :
        cytoplasm_label = np.load(cytoplasm_path)
    else : 
        cytoplasm_label = nucleus_label

    return nucleus_label, cytoplasm_label

def output_segmentation(
        path : str,
        extension : Literal['npy', 'npz_uncompressed', 'npz_compressed'],
        nucleus_label : np.ndarray,
        cytoplasm_label : np.ndarray = None,
) :
    
    saved = False
    if extension == 'npy' :
        save = np.save
    elif extension == 'npz_uncompressed' :
        save = np.savez
    elif extension == 'npz_compressed' :
        save = np.savez_compressed

    if type(nucleus_label) != type(None) :
        save(path + "_nucleus_segmentation", nucleus_label)
        saved = True
    
    if type(cytoplasm_label) != type(None) :
        save(path + "_cytoplasm_segmentation", cytoplasm_label)

    return saved