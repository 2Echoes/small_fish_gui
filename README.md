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

You are all set! Try it yourself or check the [get started](https://github.com/2Echoes/small_fish_gui/wiki/Get-started) section in the wiki.

## Developpement

Optional features to include in future versions : 

**Major Dev**
* 3D segmentation (coming soon with cellpose 4.x)