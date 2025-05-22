"""
Submodule containing custom class for napari widgets
"""
import numpy as np
import pandas as pd
import bigfish.detection as detection
from napari.layers import Labels, Points
from magicgui import magicgui

from abc import ABC, abstractmethod
from typing import Tuple

class NapariWidget(ABC) :
    """
    Common super class for custom widgets added to napari interface during run
    Each sub class as a specific function, but the widget can be acess with attribute .widget
    """
    def __init__(self):
        self.widget = self._create_widget()

    @abstractmethod
    def _create_widget(self) :
        """
        This should return a widget you can add to the napari (QWidget)
        """
        pass

class ClusterWidget(NapariWidget) :
    """
    Widget for clusters interaction are all init with cluster_layer and single_layer
    """
    def __init__(self, cluster_layer : Points, single_layer : Points):
        self.cluster_layer = cluster_layer
        self.single_layer = single_layer
        super().__init__()

class ClusterWizard(ABC) :
    """
    Common super class for all classes that will interact on single layer and cluster layer to synchronise them or modify their display.
    Their action is started through 'start_listening' method.
    To register them in CLUSTER_WIZARD they should only take single_layer and cluster_layer as arguments
    """

    def __init__(self, single_layer : Points, cluster_layer : Points):
        self.single_layer = single_layer
        self.cluster_layer = cluster_layer
        self.start_listening()
    
    def start_listening(self) :
        """
        This activate the class function. Returns None
        """
        pass


CLUSTER_WIZARDS = []
def register_cluster_wizard(cls):
    """
    Helper to register all clusters related class
    """
    CLUSTER_WIZARDS.append(cls)
    return cls

def initialize_all_cluster_wizards(single_layer: Points, cluster_layer: Points):
    """
    Initialize all wizards for cluster interaction in Napari
    """
    return [
        cls(single_layer, cluster_layer)
        for cls in CLUSTER_WIZARDS
    ]


class CellLabelEraser(NapariWidget) :
    """
    Widget for deleting cells from multiple label layers in a Napari viewer.
    """
    def __init__(self, label_list: 'list[Labels]'):
        self.label_list = label_list
        if len(self.label_list) == 0 : raise ValueError("Empty label list")
        for label_layer in self.label_list :
            label_layer.events.selected_label.connect((self, 'update'))
        super().__init__()

    def update(self, event) :
        layer : Labels = event.source
        new_label = layer.selected_label
        self.widget.label_number.value = new_label
        self.widget.update()
    
    def _create_widget(self) :
        @magicgui(
                call_button="Delete cell",
                auto_call=False
                )
        def label_eraser(label_number: int) -> None :

            for i, label in enumerate(self.label_list) :
                self.label_list[i].data[label.data == label_number] = 0
                label.refresh()

        return label_eraser


class FreeLabelPicker(NapariWidget) :
    """
    This widget gives the user a free label number
    """
    def __init__(self, label_list : 'list[Labels]'):
        self.label_list = label_list
        if len(self.label_list) == 0 : raise ValueError("Empty label list")
        super().__init__()
    
    def _create_widget(self) :
        @magicgui(
            call_button="Pick free label",
            auto_call=False
        )
        def label_pick()->None :
            max_list = [label_layer.data.max() for label_layer in self.label_list]
            new_label = max(max_list) + 1
            for label_layer in self.label_list :
                label_layer.selected_label = new_label
                label_layer.refresh()

        return label_pick


class SegmentationReseter(NapariWidget) :
    """
    This widget reset the segmentation mask as it used to be when iniating the instance
    """
    def __init__(self, label_list: 'list[Labels]'):
        self.label_list = label_list
        if len(self.label_list) == 0 : raise ValueError("Empty label list")
        self.save = self._get_save()
        super().__init__()
        
    
    def _get_save(self, label_list : 'list[Labels]') :
        return [label.data.copy() for label in label_list]

    def _create_widget(self) :
        @magicgui(
            call_button= 'Reset segmentation',
            auto_call=False,
        )
        def reset_segmentation() -> None:
            for save_data, layer in zip(self.save, self.label_list) :
                layer.data = save_data.copy()
                layer.refresh()

        return reset_segmentation

class ChangesPropagater(NapariWidget) :
    """
    Apply the changes across the vertical direction (Zstack) if confling values are found for a pixel, max label is kept.
    """
    def __init__(self, label_list):
        self.label_list = label_list
        super().__init__()

    def _create_widget(self) :
        @magicgui(
            call_button='Apply changes',
            auto_call=False,
        )
        def apply_changes() -> None:
            for layer in self.label_list :
                slices = layer.data.shape[0]
                layer_2D = np.max(layer.data, axis=0)
                layer.data = np.repeat(layer_2D[np.newaxis], slices, axis=0)
                layer.refresh()
        return apply_changes

class ClusterIDSetter(ClusterWidget) :
    """
    Allow user to set selected single spots to chosen cluster_id
    """
    def __init__(self, single_layer : Points, cluster_layer : Points):
        super().__init__(cluster_layer, single_layer)

    def _create_widget(self):

        @magicgui(
                call_button= "Set cluster ID",
                auto_call= False,
                cluster_id= {'min' : -1},
        )
        def set_cluster_id(cluster_id : int) :
            if cluster_id == -1 or cluster_id in self.cluster_layer.features['cluster_id'] :
                spots_selection = list(self.single_layer.selected_data)
                cluster_id_in_selection = list(self.single_layer.features.loc[spots_selection,["cluster_id"]].to_numpy().flatten()) + [cluster_id]
                self.single_layer.features.loc[spots_selection,["cluster_id"]] = cluster_id

                for cluster_id in np.unique(cluster_id_in_selection): # Then update number of spots in cluster
                    if cluster_id == -1 : continue
                    new_spot_number = len(self.single_layer.features.loc[self.single_layer.features['cluster_id'] == cluster_id])
                    self.cluster_layer.features.loc[self.cluster_layer.features['cluster_id'] == cluster_id, ["spot_number"]] = new_spot_number
                self.cluster_layer.events.features()

            self.cluster_layer.selected_data.clear()

        return set_cluster_id

class ClusterMerger(ClusterWidget) :
    """
    Merge all selected clusters by replacing cluster ids of all clusters and belonging points with min for cluster id.
    """
    def __init__(self, cluster_layer, single_layer):
        super().__init__(cluster_layer, single_layer)
    
    
    def _create_widget(self):

        @magicgui(
            call_button="Merge Clusters",
            auto_call=False
        )
        def merge_cluster()-> None :
            selected_clusters = list(self.cluster_layer.selected_data)
            if len(selected_clusters) == 0 : return None

            selected_cluster_ids = self.cluster_layer.features.loc[selected_clusters,['cluster_id']].to_numpy().flatten()
            new_cluster_id = selected_cluster_ids.min()

            #Dropping selected clusters
            self.cluster_layer.data = np.delete(self.cluster_layer.data, selected_clusters, axis=0)

            #Updating spots
            belonging_spots = self.single_layer.features.loc[self.single_layer.features['cluster_id'].isin(selected_cluster_ids)].index
            self.single_layer.features.loc[belonging_spots, ["cluster_id"]] = new_cluster_id

            #Creating new cluster
            centroid = list(self.single_layer.data[belonging_spots].mean(axis=0).round().astype(int))
            spot_number = len(belonging_spots)
            self.cluster_layer.data = np.append(
                self.cluster_layer.data,
                [centroid],
                axis=0
            )

            last_index = len(self.cluster_layer.data) - 1
            self.cluster_layer.features.loc[last_index, ['cluster_id']] = new_cluster_id
            self.cluster_layer.features.loc[last_index, ['spot_number']] = spot_number

            self.cluster_layer.selected_data.clear()
            self.cluster_layer.refresh()

        return merge_cluster




class ClusterUpdater(NapariWidget) :
    """
    Relaunch clustering algorithm taking into consideration new spots, new clusters and deleted clusters.
    """
    def __init__(
            self, 
            single_layer : Points, 
            cluster_layer : Points, 
            default_cluster_radius : int, 
            default_min_spot : int,
            voxel_size : 'tuple[int]'
            ):
        self.single_layer = single_layer
        self.cluster_layer = cluster_layer
        self.cluster_radius = default_cluster_radius
        self.min_spot = default_min_spot
        self.voxel_size = voxel_size
        super().__init__()

    def _compute_clusters(
            self, 
            cluster_radius : int, 
            min_spot : int
            ) -> Tuple[np.ndarray, np.ndarray, dict, dict] :
        """
        Compute clusters using bigfish detection.detect_clusters and seperate coordinates from features.
        """
        
        clustered_spots, clusters = detection.detect_clusters(
            voxel_size=self.voxel_size,
            spots= self.single_layer.data,
            radius=cluster_radius,
            nb_min_spots= min_spot
        )

        clusters_coordinates = clusters[:,:-2]
        clusters_features = {
            "spot_number" : clusters[:,-2],
            "cluster_id" : clusters[:,-1],
        }

        spots_coordinates = clustered_spots[:,:-1]
        spots_features = {
            "cluster_id" : clustered_spots[:,-1]
        }

        return clusters_coordinates, spots_coordinates, clusters_features, spots_features

    def _update_layers(
            self, 
            clusters_coordinates : np.ndarray, 
            spots_coordinates : np.ndarray, 
            clusters_features : dict, 
            spots_features : dict
            ) -> None  :
        """
        Update Points layers inside napari viewer.
        """
        
        #Modify layers
        self.single_layer.data = spots_coordinates
        self.cluster_layer.data = clusters_coordinates
        self.single_layer.features.loc[:,["cluster_id"]] = spots_features['cluster_id']
        self.cluster_layer.features.loc[:,["cluster_id"]] = clusters_features['cluster_id']
        self.cluster_layer.features.loc[:,["spot_number"]] = clusters_features['spot_number']

        self.cluster_layer.selected_data.clear()
        self.single_layer.refresh()
        self.cluster_layer.refresh()

        

    def _create_widget(self):

        @magicgui(
                call_button= "Relaunch Clustering",
                auto_call= False
        )
        def relaunch_clustering(
            cluster_radius : int = self.cluster_radius,
            min_spot : int = self.min_spot,
        ) :
            clusters_coordinates, spots_coordinates, clusters_features, spots_features = self._compute_clusters(cluster_radius=cluster_radius, min_spot=min_spot)
            self._update_layers(clusters_coordinates, spots_coordinates, clusters_features, spots_features )
            self.cluster_radius = cluster_radius
            self.min_spot = min_spot

        return relaunch_clustering

class ClusterCreator(ClusterWidget) :
    """
    Create a cluster containing all and only selected spots located at the centroid of selected points.
    """
    def __init__(self, cluster_layer, single_layer):
        super().__init__(cluster_layer, single_layer)

    def _create_widget(self):

        @magicgui(
                call_button= "Create Cluster",
                auto_call=False
        )
        def create_foci() -> None :
            selected_spots_idx = pd.Index(list(self.single_layer.selected_data))
            free_spots_idx : pd.Index = self.single_layer.features.loc[self.single_layer.features['cluster_id'] == -1].index
            selected_spots_idx = selected_spots_idx[selected_spots_idx.isin(free_spots_idx)]

            spot_number = len(selected_spots_idx)
            if spot_number == 0 :
                print("To create a cluster please select at least 1 spot")
            else :
                
                #Foci creation
                spots_coordinates = self.single_layer.data[selected_spots_idx]
                new_cluster_id = self.cluster_layer.features['cluster_id'].max() + 1
                centroid = list(spots_coordinates.mean(axis=0).round().astype(int))

                self.cluster_layer.data = np.concatenate([
                    self.cluster_layer.data,
                    [centroid]
                ], axis=0)
                
                last_index = len(self.cluster_layer.data) - 1
                self.cluster_layer.features.loc[last_index, ['cluster_id']] = new_cluster_id
                self.cluster_layer.features.loc[last_index, ['spot_number']] = spot_number

                #Update spots cluster_id
                self.single_layer.features.loc[selected_spots_idx,["cluster_id"]] = new_cluster_id
        
        return create_foci

@register_cluster_wizard
class ClusterInspector :
    """
    Listen to event on cluster layer to color spots belonging to clusters in green
    """
    def __init__(self, single_layer : Points, cluster_layer : Points):
        self.single_layer = single_layer
        self.cluster_layer = cluster_layer
        self.start_listening()

    def reset_single_colors(self) -> None:
        self.single_layer.face_color = [0,0,0,0] #transparent
        self.single_layer.refresh()

    def start_listening(self) :

        def color_single_molecule_in_foci() -> None:
            self.reset_single_colors()
            selected_cluster_indices = self.cluster_layer.selected_data
            for idx in selected_cluster_indices :
                selected_cluster = self.cluster_layer.features.at[idx,"cluster_id"]
                belonging_single_idex = self.single_layer.features.loc[self.single_layer.features['cluster_id'] == selected_cluster].index.to_numpy()
                self.single_layer.face_color[belonging_single_idex] = [0,1,0,1] #Green
                self.single_layer.refresh()

        self.cluster_layer.selected_data.events.items_changed.connect(color_single_molecule_in_foci)

@register_cluster_wizard
class ClusterEraser(ClusterWizard) :
    """
    When a foci is deleted, update spots feature table accordingly.
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)

    def start_listening(self):
        self.original_remove_selected = self.cluster_layer.remove_selected
    
        def remove_selected_cluster() :
            selected_cluster = self.cluster_layer.selected_data
            for cluster_idx in selected_cluster : #First we update spots data
                cluster_id = self.cluster_layer.features.at[cluster_idx, "cluster_id"]
                self.single_layer.features.loc[self.single_layer.features['cluster_id'] == cluster_id, ['cluster_id']] = -1
            
            self.original_remove_selected() # Then we launch the usual napari method
        
        self.cluster_layer.remove_selected = remove_selected_cluster

@register_cluster_wizard
class ClusterAdditionDisabler(ClusterWizard) :
    """
    Remove the action when user uses points addition tool for Foci, forcing him to use the FociCreator tool to add new cluster.
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)
    
    def start_listening(self):

        def print_excuse(*args, **kwargs):
            print("Spot addition is disabled for cluster layer. Use the foci creation tool below after selecting spots you want to cluster")

        self.cluster_layer.add = print_excuse

@register_cluster_wizard
class SingleEraser(ClusterWizard) :
    """
    When a single is deleted, update clusters feature table accordingly
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)

    def start_listening(self):
        self._origin_remove_single = self.single_layer.remove_selected

        def delete_single(*args, **kwargs) :
            selected_single_idx = list(self.single_layer.selected_data)
            modified_cluster_ids = self.single_layer.features.loc[selected_single_idx, ["cluster_id"]].to_numpy().flatten()

            print(np.unique(modified_cluster_ids, return_counts=True))
            for cluster_id, count in zip(*np.unique(modified_cluster_ids, return_counts=True)): # Then update number of spots in cluster
                    if cluster_id == -1 : continue
                    new_spot_number = len(self.single_layer.features.loc[self.single_layer.features['cluster_id'] == cluster_id]) - count #minus number of spot with this cluster id we remove
                    print("new spot number : ", new_spot_number)
                    print('target cluster id : ', cluster_id)
                    self.cluster_layer.features.loc[self.cluster_layer.features['cluster_id'] == cluster_id, ["spot_number"]] = new_spot_number
            self._origin_remove_single()
            self.cluster_layer.events.features()
        
        self.single_layer.remove_selected = delete_single


@register_cluster_wizard
class ClusterCleaner(ClusterWizard) :
    """
    Deletes clusters if they drop to 0 single molecules.
    """

    def __init__(self, single_layer, cluster_layer):
        super().__init__(single_layer, cluster_layer)

    def start_listening(self):

        def delete_empty_cluster() :
            drop_idx = self.cluster_layer.features[self.cluster_layer.features['spot_number'] == 0].index
            print("drop_idx : ",drop_idx)
            
            if len(drop_idx) > 0 :
                print("Removing {} empty cluster(s)".format(len(drop_idx)))
                self.cluster_layer.data = np.delete(self.cluster_layer.data, drop_idx, axis=0)
                self.cluster_layer.refresh()

        self.cluster_layer.events.features.connect(delete_empty_cluster)
        