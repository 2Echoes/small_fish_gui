import os
import pandas as pd
from bigfish.stack import check_parameter

def save_results(dataframes: 'list[pd.DataFrame]', path:str, filename:str, do_excel= True, do_feather= False) :
    check_parameter(dataframes= list, path= str, filename = str, do_excel = bool, do_feather = bool)

    if not path.endswith('/') : path +='/'
    assert os.path.isdir(path)

    for idx, dataframe in enumerate(dataframes) :
        check_parameter(dataframe= pd.DataFrame)
        filename = '{1}_{0}'.format(idx, filename)
        if do_excel : dataframe.to_excel(path + filename + '.xlsx')
        if do_feather : dataframe.to_feather(path + filename + '.feather')

    return True