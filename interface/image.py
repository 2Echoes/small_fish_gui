from bigfish.stack import read_image
from czifile import imread
from ..utils import check_parameter
import re

class FormatError(Exception):
    pass


def open_image(full_path:str) :
    if full_path.endswith('.czi') : im = imread(full_path)
    else : im = read_image(full_path)

    reshape = []
    for axis in im.shape :
        if axis != 1 : reshape.append(axis)
    im = im.reshape(reshape)

    return im


def check_format(image, is_3D, is_time_stack, is_multichannel) :
    shape = list(image.shape)
    dim = image.ndim - (shape[image.ndim - 1] == 1)
    if not dim == (2 + is_3D + is_time_stack + is_multichannel) :
        raise FormatError("Inconsistency in image format and parameters.")



def get_filename(full_path: str) :
    check_parameter(full_path=str)

    pattern = r'.*\/(.+)\..*$'
    if not full_path.startswith('/') : full_path = '/' + full_path
    re_match = re.findall(pattern, full_path)
    if len(re_match) == 0 : raise ValueError("Could not read filename from image full path.")
    if len(re_match) == 1 : return re_match[0]
    else : raise AssertionError("Several filenames read from path")