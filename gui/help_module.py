import PySimpleGUI as sg
import os


def _fake_help() :
    layout = [
        [sg.Text("Fake help window")],
        [sg.Button('Close')]
    ]

    return layout

def ask_help(chapter= '') :
    
    if chapter == 'general' :
        help_l = _general_help()

    elif chapter == 'segmentation' :
        help_l= _segmentation_help()

    elif chapter == 'mapping' : 
        help_l = _mapping_help()

    elif chapter == 'detection' :
        help_l = _detection_help()

    else : 
        help_l = _fake_help()

    window = sg.Window('Help (small fish)', layout=help_l, keep_on_top=True, auto_size_text=True)
    event, values = window.read(timeout= 0.1)

    if event == 'Close' :
        window.close()

def add_help_button(help_request) :
    pass

def _general_help() :

    im_path = os.path.dirname(__file__) + '/general_help_screenshot.png'

    help_text = """
    Pipeline settings :

        Dense regions deconvolution : (Recommanded for cluster computations) Detect dense and bright regions with potential clustered 
            spots and simulate a more realistic number of spots in these regions.
            See bigfish documentation : https://big-fish.readthedocs.io/en/stable/detection/dense.html

        Cluster computation :
            DBScan algorithm is ran by big-fish to detecte clusters of spots. Use is you want to quantify one of the following : 
            Transcription sites, foci, colocalisation of spots near foci...
        
        Segmentation : Perform full cell segmentation in 2D (cytoplasm + nucleus) via cellpose.
            You can use your own retrained models or out of the box models from cellpose.

        Napari correct :
            After each detection, opens a Napari viewer, enabling the user to visualise, add or remove spots and clusters.
    """

    layout = [
        [sg.Text("Welcome to small fish", font= 'bold 15')],
        [sg.Image(im_path)],
        [sg.Text(help_text, font = 'bold 10')]
    ]

    return layout

def _detection_help() :

    header = """
    Detection is the main feature we use from big-fish package (usage requires quote).
    Access fully detailed documentation : 
    
        DETECTION : https://big-fish.readthedocs.io/en/stable/detection/spots.html
        DENSE REGIONS DECONVOLUTION : https://big-fish.readthedocs.io/en/stable/detection/dense.html
        CLUSTERING : https://big-fish.readthedocs.io/en/stable/detection/cluster.html
        
    """
    detection_header= """
    
    DETECTION PARAMETERS
    """
    detection_text = """
    threshold
        Leave empty for automatic threshold computation (see doc). Or set a manual threshold to apply after LoG filter.
    
    threshold penalty
        Custom feature. Apply a multiplicator to automatic threshold.
        Leave empty or 1 for no modification of auto threshold.
        From 0<.<1 values will lower the threshold, increasing the number of spot detected.
        On the contrary >1 values will increase the threshold, reducing the number of spot detected.
             
    voxel_size
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or yx dimensions). 
        If it’s a scalar, the same value is applied to every dimensions. Not used if ‘log_kernel_size’ and ‘minimum_distance’ are provided.

    spot_size    
        Radius of the spot, in nanometer. One value per spatial dimension (zyx or yx dimensions). 
        If it’s a scalar, the same radius is applied to every dimensions. 
        Not used if ‘log_kernel_size’ and ‘minimum_distance’ are provided.

    log kernel size
        Size of the LoG kernel. It equals the standard deviation (in pixels) used for the gaussian kernel 
        (one for each dimension). One value per spatial dimension (zyx or yx dimensions). If it’s a scalar, 
        the same standard deviation is applied to every dimensions. 
        If None, we estimate it with the voxel size and spot radius.

    minimum distance
        Minimum distance (in pixels) between two spots we want to be able to detect separately. 
        One value per spatial dimension (zyx or yx dimensions). 
        If it’s a scalar, the same distance is applied to every dimensions. 
        If None, we estimate it with the voxel size and spot radius.

    """
    deconv_header="""    
    DENSE REGIONS DECONVOLUTION PARAMETERS

    """
    deconv_text="""
    alpha
        Note : Simply put the higher alpha the less spots are added in bright regions.

        Intensity percentile used to compute the reference spot, between 0 and 1. 
        The higher, the brighter are the spots simulated in the dense regions. 
        Consequently, a high intensity score reduces the number of spots added. 
        Default is 0.5, meaning the reference spot considered is the median spot.
    
    beta
        Note : Simply put the higher beta the brighter regions need to be to be deconvoluted.

        Multiplicative factor for the intensity threshold of a dense region. Default is 1. 
        Threshold is computed with the formula:

                threshold = beta * max(median_spot)

        with median_spot the median value of all detected spot signals.
    
    gamma
        Note : for gamma = 0 no gaussian filter is performed.

        Multiplicative factor use to compute the gaussian kernel size:

                kernel_size = gamma * spot_size / voxel_size

        We perform a large gaussian filter with such scale to estimate image background and remove it from original image. 
        A large gamma increases the scale of the gaussian filter and smooth the estimated background. 
        To decompose very large bright areas, a larger gamma should be set.

    kernel_size
        Standard deviation used for the gaussian kernel (one for each dimension), in pixel. 
        If it’s a scalar, the same standard deviation is applied to every dimensions. 
        If None, we estimate the kernel size from ‘spot_radius’, ‘voxel_size’ and ‘gamma’


    """
    clustering_header="""    
    CLUSTERING PARAMETERS

    """
    clustering_text="""
    cluster size
        The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        Radius expressed in nanometer.
        
        NOTE : Cluster centroids are computed from bigfish DBScan algorithm. 
        But the number of spots belonging to those clusters is computed as the number of sptos closer than the cluster_size distance (nanometer).
        Which can yield a slightly different result than bigfish but allow us to add and delete cluster with the napari correction option.

    min number of spots
        The number of spots in a neighborhood for a point to be considered as a core point (from which a cluster is expanded). 
        This includes the point itself.

    """

    quote = """
    BigFish quote : 
    Arthur Imbert, Wei Ouyang, Adham Safieddine, Emeline Coleno, Christophe Zimmer, Edouard Bertrand, Thomas Walter, Florian Mueller. 
    FISH-quant v2: a scalable and modular analysis tool for smFISH image analysis. bioRxiv (2021) 
    https://doi.org/10.1101/2021.07.20.45302

    """

    layout = [
        [sg.Text(header, font= "bold 15")],
        [sg.Column([
            [sg.Text(detection_header, font= 'bold 13')],
            [sg.Text(detection_text)],
            [sg.Text(deconv_header, font= 'bold 13')],
            [sg.Text(deconv_text)],
            [sg.Text(clustering_header, font= 'bold 13')],
            [sg.Text(clustering_text)],
            ], scrollable=True, vertical_scroll_only=True)],
        [sg.Text(quote, font= 'italic 8')]
    ]

    return layout

def _segmentation_help() :

    cellpose1_quote = """Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
    Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106."""
    cellpose2_quote = """Pachitariu, M. & Stringer, C. (2022).
    Cellpose 2.0: how to train your own model. Nature methods, 1-8."""
    im_path = os.path.dirname(__file__) + '/segmentation_help_screenshot.png'

    layout =[
        [sg.Text("Segmentation is performed using Cellpose 2.0; this is published work that requires citation.\n")],
        [sg.Text(cellpose1_quote)],
        [sg.Text(cellpose2_quote)],
        [sg.Image(im_path)]
    ]

    return layout


def _mapping_help() :

    im_path = os.path.dirname(__file__) + '/mapping_help_screenshot.png'


    help_text = """
    Depending on image format, dimensions (time, channels, spatial dimensions) are not always stored in the same order.
    An automatic configuration is performed; nonetheless it is recommanded to check it worked properly or your data can
    get mixed up.

    This window present the shape of your image : example (1080,1080,4)
    1080x1080 are the xy dimension (pixel resolution); and 4 is the number of channel. Another example a 3D multichannel 
    stack could be (18,4,1080,1080)...
    The machine understand the order of the information such as (1080,1080,4) positions are (0,1,2). /!\ It starts from zero! /!\
    The mapping purpose is to link the position to the type of informations, in this case we want :

            x : 1
            y : 0
            z : None
            c : 2
            t : None

    The other example from above (18,4,1080,1080) mapping would be :

            x : 3
            y : 2
            z : 0
            c : 1
            t : None
    """

    layout = [
        [sg.Text(help_text)],
        [sg.Image(im_path)]
    ]

    return layout

def _small_fish_help() :
    pass
