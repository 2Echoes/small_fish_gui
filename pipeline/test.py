import small_fish.pipeline._segmentation as s
import small_fish.gui.prompts as p
import small_fish.pipeline.actions as a
import bigfish.stack as stack
import bigfish.plot as plot
import numpy as np
import PySimpleGUI as sg

image: np.ndarray = stack.read_image('/home/floricslimani/Documents/Python_scripts/small_fish/HELA_3C_3D12S_D1_clusters.tif')
# image = np.moveaxis(image,2,0)


label1, label2 = a.launch_segmentation(image)
print(label1)
print(label2)


# import itertools
# import threading
# import time
# import sys

# done = False
# #here is the animation
# def animate():
#     while True:
#         sg.popup_animated(sg.DEFAULT_BASE64_LOADING_GIF, time_between_frames=100)
#         if done:
#             break
#     #     sys.stdout.write('\rloading ' + c)
#     #     sys.stdout.flush()
#     #     time.sleep(0.1)
#     # sys.stdout.write('\rDone!     ')

# t = threading.Thread(target=animate)
# t.start()

# #long process here
# time.sleep(10)
# done = True
