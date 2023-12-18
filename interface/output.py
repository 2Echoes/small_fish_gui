import os, re
import pandas as pd
from bigfish.stack import check_parameter

def save_results(dataframes: 'list[pd.DataFrame]', path:str, do_excel= True, do_feather= False) :
    check_parameter(dataframes= list, path= path)

    if not os.path.isdir(path) :
        pattern = r'(.*\/).+\..*$'
        dir_path = re.findall(pattern,path)
        if len(dir_path) == 0 :
            dir_path = dir_path[0]
        else : raise AssertionError()
    else :
        if not path.endswith('/') : path +='/'

    assert os.path.isdir(path)

    for idx, dataframe in enumerate(dataframes) :
        check_parameter(dataframe, pd.DataFrame)
        filename = 'results_{0}'.format(idx)
        if do_excel : dataframe.to_excel(path + filename + '.xlsx')
        if do_feather : dataframe.to_feather(path + filename + '.feather')

    return True