import os
import pandas as pd
from bigfish.stack import check_parameter



def _cast_spot_to_tuple(spot) :
    return tuple([coord for coord in spot])

def _cast_spots_to_tuple(spots) :
    return tuple(list(map(_cast_spot_to_tuple, spots)))

def write_results(dataframe: pd.DataFrame, path:str, filename:str, do_excel= True, do_feather= False) :
    check_parameter(dataframe= pd.DataFrame, path= str, filename = str, do_excel = bool, do_feather = bool)

    if len(dataframe) == 0 : return True
    if not do_excel and not do_feather : 
        return False

    if not path.endswith('/') : path +='/'
    assert os.path.isdir(path)


    new_filename = filename
    i= 1
    while new_filename + '.xlsx' in os.listdir(path) or new_filename + '.feather' in os.listdir(path) :
        new_filename = filename + '_{0}'.format(i)
        i+=1

    if 'image' in dataframe.columns :
        dataframe = dataframe.drop(['image'], axis=1)

    if 'spots' in dataframe.columns : 
        dataframe = dataframe.drop(['spots'], axis= 1)
        
    if 'clusters' in dataframe.columns : 
        dataframe = dataframe.drop(['clusters'], axis= 1)

    if do_excel : dataframe.reset_index(drop=True).to_excel(path + filename + '.xlsx')
    if do_feather : dataframe.reset_index(drop=True).to_feather(path + filename + '.feather')

    return True