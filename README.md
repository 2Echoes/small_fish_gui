# Small Fish
**Small Fish** is a python application for the analysis of smFish images. It provides a ready to use graphical interface to combine famous python packages for cell analysis without any need for coding.

Cell segmentation (**2D**) is peformed using *cellpose* (published work) : https://github.com/MouseLand/cellpose; compatible with your own cellpose models.

Spot detection is performed via *big-fish* (published work) : https://github.com/fish-quant/big-fish

Time stacks are not yet supported.

## What can you do with small fish ?

- Single molecule quantification (including a lot of spatial features)
- Foci/Transcription site quantification
- Nuclear signal quantification
- Signal to noise analysis
- multichannel colocalisation

<img src="https://github.com/2Echoes/small_fish_gui/blob/main/Segmentation%20example.jpg" width="500" title="Cell segmentation with Cellpose" alt="Cell segmentation - cellpose">| <img src="https://github.com/2Echoes/small_fish_gui/blob/main/napari_detection_example.png" width="500" title="Spot detection; clustering visualisation on Napari" alt="detection; Napari example">

## Installation
If you don't have a python installation yet I would recommend the [miniconda distribution](https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/); but any distribution should work.

It is higly recommanded to create a specific [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtual](https://docs.python.org/3.6/library/venv.html) environnement to install small fish.

```bash
conda create -n small_fish python=3.8
conda activate small_fish
```
Then download the small_fish package : 
```bash
pip install small_fish_gui
```
<b> (Recommended) </b> Results visualisation is achieved through *Napari* which you can install with :

```bash
pip install napari[all]
```

## Run Small fish

First activate your python environnement : 
```bash
activate small_fish
```
Then launch Small fish : 
```bash
python -m small_fish_gui
```

## Cellpose configuration

For the following steps first activate your small fish environnement : 

```bash
conda activate small_fish
```
### Setting up your GPU for cellpose (Windows / Linux)
This instructions describe how I installed CUDA and GPU cellpose on the machines I tested, unfortunatly, drivers installations don't always run smoothly, if you run into any difficulties please have a look at the *GPU version (CUDA) on Windows or Linux* section of the [cellpose documentation](https://github.com/MouseLand/cellpose) for assistance.

First step is to check that your GPU is CUDA compatible which it should be if from the brand NVIIDA.
Then you need to install CUDA from the [NVIDIA archives](https://developer.nvidia.com/cuda-toolkit-archive), any 11.x version should work but I recommend the 11.8 version.

Finally we need to make some modifcation to your small fish environnement : 

Remove the CPU version of torch

```bash
pip uninstall torch
```
Then install pytorch and cudatoolkit :

```bash
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
```
If the installation succeeded next time your run segmentation with small fish you should see the "GPU is ON" notice upon entering the segmentation parameters.
If you run into any problems I would recommend following the official cellpose instructions as mentionned above.


### Training cellpose
If you want to train your own cellpose model or import custom model from exterior source I recommend doing so from the cellpose GUI 

To install the GUI run : 

```bash
pip install cellpose[gui]
```
Then to run cellpose
```bash
cellpose
```
Note that for training it is recommended to first set up your GPU as training computation can be quite long otherwise. To get started with how to train your models you can watch the [video](https://www.youtube.com/watch?v=5qANHWoubZU) from cellpose authors.

## Developpement

Optional features to include in future versions : 
- batch processing
- time stack (which would include cell tracking)
- 3D segmentation
