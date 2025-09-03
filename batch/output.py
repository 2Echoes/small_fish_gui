import numpy as np
import os

def output_masks(
        output_path :str,
        nucleus_label : np.ndarray,
        cytoplasm_label : np.ndarray = None,
) :
    np.save(output_path + "_nucleus.npy", arr= nucleus_label)
    if type(cytoplasm_label) == type(None) :
        np.save(output_path + "_cytoplasm.npy", arr= nucleus_label)
