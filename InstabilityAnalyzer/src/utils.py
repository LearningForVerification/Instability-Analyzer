import copy
import os

import matplotlib.pyplot as plt
from pynever.strategies.bounds_propagation.bounds import VerboseBounds
import torch

class SmartDict:
    def __init__(self, n_hidden_layers, n_samples):
        self.unstable_neurons_number = torch.zeros(n_samples, n_hidden_layers)
        self.counter = 0
        self.n_samples = n_samples

    def update(self, new_data: list):
        to_append = torch.tensor(new_data)
        self.unstable_neurons_number[self.counter, :] = to_append
        self.counter += 1

    def get_average(self):
        assert self.counter == self.n_samples
        return (self.unstable_neurons_number.sum() / (self.counter + 1)).item()

def get_n_hidden_layers(bounds_dict: VerboseBounds):
    return len(bounds_dict.identifiers) - 1

def get_labels(bounds_dict: VerboseBounds):
    labels = bounds_dict.identifiers[:-1]
    return labels

def count_unstable_neurons(bounds_dict: VerboseBounds):
    """
    Return a list containing the number of unstable neurons for each hidden layer.
    """

    bounds_dict = copy.deepcopy(bounds_dict)

    ret_list = []

    # Delete output layers numeric bounds
    output_layer_identifier = bounds_dict.identifiers.pop(-1)
    numeric_bounds = bounds_dict.numeric_post_bounds
    del numeric_bounds[output_layer_identifier]

    # Count the number of unstable neurons per layer

    for layer_id in bounds_dict.identifiers:
        lower_bounds = numeric_bounds[layer_id].lower
        upper_bounds = numeric_bounds[layer_id].upper
        bool_mask = (lower_bounds < 0) & (upper_bounds > 0)
        true_count = bool_mask.sum().item()
        ret_list.append(true_count)

    return ret_list

def get_activation_hot_map(bounds_dict: VerboseBounds):
    """
    Return a list containing the hot maps representing the active neurons for each hidden layer.
    """

    bounds_dict = copy.deepcopy(bounds_dict)

    unstable_hot_maps = list()
    positive_stable_hot_maps = list()
    negative_stable_hot_maps = list()


    # Delete output layers numeric bounds
    output_layer_identifier = bounds_dict.identifiers.pop(-1)
    numeric_bounds = bounds_dict.numeric_post_bounds
    del numeric_bounds[output_layer_identifier]

    for layer_id in bounds_dict.identifiers:
        lower_bounds = numeric_bounds[layer_id].lower
        upper_bounds = numeric_bounds[layer_id].upper
        unstable_hot_map = (lower_bounds < 0) & (upper_bounds > 0)
        positive_stable_hot_map = (lower_bounds > 0) & (upper_bounds > 0)
        negative_stable_hot_map = (lower_bounds < 0) & (upper_bounds < 0)

        unstable_hot_maps.append(unstable_hot_map)
        positive_stable_hot_maps.append(positive_stable_hot_map)
        negative_stable_hot_maps.append(negative_stable_hot_map)

    return unstable_hot_maps, positive_stable_hot_maps, negative_stable_hot_maps


def plot_frequency_graphs(frequency_maps, limit=None, title=None, output_folder=None):
    """
    Plotta grafici delle frequenze.

    Args:
        frequency_maps (list): Lista di liste, ciascuna contenente i valori da plottare.
        limit (int, optional): Se specificato, viene tracciata una linea orrizontale a questa posizione.
        output_folder (str, optional): Se specificato, i grafici vengono salvati in questa cartella.
            Altrimenti, vengono mostrati a video.
    """
    for i, frequency_map in enumerate(frequency_maps, start=1):
        x_labels = list(range(1, len(frequency_map) + 1))
        plt.figure(figsize=(8, 6))
        plt.plot(x_labels, frequency_map, marker='o', linestyle='-', color='b')
        plt.xlabel("Neurons")
        plt.ylabel("Values of x")
        plt.xticks(x_labels)
        plt.grid(True)

        if title is not None:
            plt.title(title)
        else:
            plt.title(f"{len(frequency_map)} frequency graph")

        if limit is not None:
            plt.axhline(y=limit, color='r', linestyle='--', linewidth=1)

        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            file_path = os.path.join(output_folder, f"frequency_graph_{i}.png")
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

