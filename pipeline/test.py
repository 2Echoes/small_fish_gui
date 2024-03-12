import small_fish.pipeline._segmentation as s
import small_fish.gui.prompts as p
import small_fish.pipeline.actions as a
import bigfish.stack as stack
import bigfish.plot as plot
import numpy as np
import PySimpleGUI as sg

image: np.ndarray = stack.read_image('/home/flo/Downloads/small_fish/HEK_4C_2D_D1.tif')
image = np.moveaxis(image,2,0)


sg.Print(a.launch_segmentation(image))
# label1, label2 = a.launch_segmentation(image)
# print(label1)
# print(label2)