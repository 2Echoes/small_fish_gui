from small_fish.pipeline.napari_wrapper import correct_spots
import numpy as np


im = np.zeros((5,300,300), dtype= int)
Y,X = np.arange(300), np.arange(300)
im[:, Y, X] = 255
print(im.dtype)
points_num = 50
points_Z = np.random.randint(0,im.shape[0], points_num)
points_Y = np.random.randint(0,im.shape[1], points_num)
points_X = np.random.randint(0,im.shape[2] ,points_num)

spots = np.array(
    list(zip(points_Z, points_Y, points_X))
, dtype= int)

spots = [np.array(list(zip(
    np.random.randint(0,im.shape[0], points_num), np.random.randint(0,im.shape[1], points_num), np.random.randint(0,im.shape[2], points_num)
)), dtype= int) for i in range(11)]
corr_spots = correct_spots(im, spots)

for idx, spots in enumerate(corr_spots) : 
    print("layer {1} spots number : {0}".format(len(spots), idx+1))