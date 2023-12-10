from small_fish.gui.prompt import *
from small_fish.pipeline._preprocess import *
values = ask_input_parameters()
values = convert_parameters_types(values)
values = check_integrity(values)
del values['image']
print(values)