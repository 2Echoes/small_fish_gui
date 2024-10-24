"""
Submodule containing custom class for napari widgets
"""
import numpy as np
from napari.layers import Labels
from magicgui import magicgui

class cell_label_eraser :
    """
    Must be instanced within Napari Viewer definition range for update connection to work, cell deletion works fine anyway.
    """
    def __init__(self, label_list: 'list[Labels]'):
        self.widget = self._create_eraser(label_list)
        for label_layer in label_list :
            label_layer.events.selected_label.connect((self, 'update'))

    def update(self, event) :
        layer : Labels = event.source
        new_label = layer.selected_label
        self.widget.label_number.value = new_label
        self.widget.update()
    
    def _create_eraser(self, label_list: 'list[Labels]') :
        @magicgui(
                call_button="Delete cell",
                auto_call=False
                )
        def label_eraser(label_number: int) -> None :

            for i, label in enumerate(label_list) :
                label_list[i].data[label.data == label_number] = 0
                label.refresh()

        return label_eraser



class free_label_picker :
    def __init__(self, label_list):
        self.widget = self._create_free_label_picker(label_list)
    
    def _create_free_label_picker(self, label_list : 'list[Labels]') :
        @magicgui(
            call_button="Pick free label",
            auto_call=False
        )
        def label_pick()->None :
            max_list = [label_layer.data.max() for label_layer in label_list]
            new_label = max(max_list) + 1
            for label_layer in label_list :
                label_layer.selected_label = new_label
                label_layer.refresh()

        return label_pick


class segmentation_reseter :
    def __init__(self, label_list):
        self.save = self._get_save(label_list)
        self.widget = self._create_widget(label_list)
        
    
    def _get_save(self, label_list : 'list[Labels]') :
        return [label.data.copy() for label in label_list]

    def _create_widget(self, label_list: 'list[Labels]') :
        @magicgui(
            call_button= 'Reset segmentation',
            auto_call=False,
        )
        def reset_segmentation() -> None:
            for save_data, layer in zip(self.save, label_list) :
                layer.data = save_data.copy()
                layer.refresh()

        return reset_segmentation

class changes_propagater :
    def __init__(self, label_list):
        self.widget = self._create_widget(label_list)

    def _create_widget(self, label_list: 'list[Labels]') :
        @magicgui(
            call_button='Apply changes',
            auto_call=False,
        )
        def apply_changes() -> None:
            for layer in label_list :
                slices = layer.data.shape[0]
                layer_2D = np.max(layer.data, axis=0)
                layer.data = np.repeat(layer_2D[np.newaxis], slices, axis=0)
                layer.refresh()
        return apply_changes
