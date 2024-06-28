import PySimpleGUI as sg
from ..pipeline._preprocess import _auto_map_channels
from ..pipeline._preprocess import _check_channel_map_integrity
from ..pipeline._preprocess import MappingError

def get_elmt_from_key(Tab_elmt:sg.Tab, key) -> sg.Element:
    elmt_list = sum(Tab_elmt.Rows,[])
    for elmt in elmt_list :
        if elmt.Key == key : return elmt
    raise KeyError("{0} key not found amongst {1}.".format(key, [elmt.Key for elmt in elmt_list]))

def call_auto_map(
        tab_elmt: sg.Tab,
        shape,
        is_3D,
        is_multichannel,
    ) :
    
    #Get auto map
    try :
        map = _auto_map_channels(
            is_3D_stack=is_3D,
            is_time_stack=False,
            multichannel=is_multichannel,
            image=None,
            shape=shape
            )
    
    except MappingError as e :
        sg.popup_error(e)
        return {}
    
    else :
        #Acess elemnt
        x_elmt = get_elmt_from_key(tab_elmt, 'x')
        y_elmt = get_elmt_from_key(tab_elmt, 'y')
        z_elmt = get_elmt_from_key(tab_elmt, 'z')
        c_elmt = get_elmt_from_key(tab_elmt, 'c')

        #Update values
        x_elmt.update(value=map.get('x'))
        y_elmt.update(value=map.get('y'))
        z_elmt.update(value=map.get('z'))
        c_elmt.update(value=map.get('c'))

        return map

def create_map(
        values:dict,
        is_3D:bool,
        is_multichannel:bool
        ) :
    
    maping ={
        'x': values.get('x'),
        'y': values.get('y')
    }

    if is_3D : maping['z'] = values.get('z')
    if is_multichannel : maping['c'] = values.get('c')
    
    return maping

