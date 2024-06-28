"""
Submodule handling all parameters check, asserting functions and pipeline will be able to run.
"""

import czifile as czi
import bigfish.stack as stack
import numpy as np
import PySimpleGUI as sg

def check_file(filename:str) :

    if filename.endswith('.czi') :
        image = czi.imread(filename)
    else :
        image = stack.read_image(filename)

    image = np.squeeze(image)

    return image.shape

def sanity_check(
        filename_list: list, 
        batch_folder : str,
        window : sg.Window, 
        progress_bar: sg.ProgressBar,
        ) :
    
    filenumber = len(filename_list)
    if filenumber == 0 :
        print("No file to check")
        progress_bar.update(current_count= 0, bar_color=('gray','gray'))
        return None
    else :
        print("{0} files to check".format(filenumber))
        progress_bar.update(current_count=0, max= filenumber)
        ref_shape = check_file(batch_folder + '/' + filename_list[0])

        print("Starting sanity check. This could take some time...")
        for i, file in enumerate(filename_list) :
            progress_bar.update(current_count= i+1, bar_color=('green','gray'))
            shape = check_file(batch_folder + '/' + file)

            if len(shape) != len(ref_shape) : #then dimension missmatch
                print("Different number of dimensions found : {0}, {1}".format(len(ref_shape), len(shape)))
                progress_bar.update(current_count=filenumber, bar_color=('red','black'))
                window= window.refresh()
                break

            window= window.refresh()

        print("Sanity check completed.")
        return None if len(shape) != len(ref_shape) else shape
