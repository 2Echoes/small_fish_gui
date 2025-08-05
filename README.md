# Small Fish - A User-Friendly Graphical Interface for smFISH Image Quantification

**Small Fish** is a python application for the analysis of smFish images. It provides a ready to use graphical interface to synthetize state-of-the-art scientific packages into an automated workflow. Small Fish is designed to simplify images quantification and analysis for people without coding skills. 

Cell segmentation is peformed in 2D using *cellpose*; (**cellpose 3D will be available soon**) (published work) : https://github.com/MouseLand/cellpose; compatible with your own cellpose models.

Spot detection is performed via *big-fish* (published work) : https://github.com/fish-quant/big-fish, the python implementation of **Fish-quant**.

***The workflow is fully explained in the [wiki](https://github.com/2Echoes/small_fish_gui/wiki) ! Make sure to check it out.***

## What can you do with small fish ?

- Single molecule quantification
- Transcriptomics
- Foci/Transcription site quantification
- Nuclear signal quantification
- Signal to noise analysis
- Cell segmentation
- Multichannel colocalisation

<img src="https://github.com/2Echoes/small_fish_gui/blob/main/Segmentation%20example.jpg" width="500" title="Cell segmentation with Cellpose" alt="Cell segmentation - cellpose">| <img src="https://github.com/2Echoes/small_fish_gui/blob/main/napari_detection_example.png" width="500" title="Spot detection; clustering visualisation on Napari" alt="detection; Napari example">

## Installation
If you don't have a python installation yet I would recommend the [miniconda distribution](https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/); but any distribution should work.

It is higly recommanded to create a specific [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtual](https://docs.python.org/3.6/library/venv.html) environnement to install small fish.

```bash
conda create -n small_fish python=3.9
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


### Training cellpose
If you want to train your own cellpose model or import custom model from exterior source I recommend doing so from the cellpose GUI. Note that Small fish uses mean projection to segment images in 2D, if you want to retrain a cellpose model to fit your data it is recommended to do so on mean or max projection of your data.

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

**Major Dev**
* 3D segmentation (coming soon with cellpose 4.x)