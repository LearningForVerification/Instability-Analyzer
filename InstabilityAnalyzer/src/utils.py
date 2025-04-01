import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from pynever.strategies.bounds_propagation.bounds import VerboseBounds


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
        plt.bar(x_labels, frequency_map, color='b',width=0.7)
        plt.xlabel(f"Neurons of a {len(frequency_map)} layer")
        plt.ylabel("Values of x")
        #plt.xticks(x_labels)
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

def plot_multiple_frequency_graphs(frequency_maps, limit=None, title=None, output_folder=None, log_scale=False, cols=2):
    """
    Plotta più grafici delle frequenze in un'unica immagine.

    Args:
        frequency_maps (list): Lista di liste, ciascuna contenente i valori da plottare.
        limit (int, optional): Se specificato, traccia una linea orizzontale a questa posizione.
        output_folder (str, optional): Se specificato, salva l'immagine in questa cartella.
        cols (int, optional): Numero di colonne nei subplot.
    """
    num_graphs = len(frequency_maps)
    rows = (num_graphs + cols - 1) // cols  # Calcola il numero di righe

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))  # Layout dinamico
    axes = axes.flatten()  # Converte in lista per iterare più facilmente

    for i, (ax, frequency_map) in enumerate(zip(axes, frequency_maps)):
        x_labels = list(range(1, len(frequency_map) + 1))
        ax.bar(x_labels, frequency_map, color='b', alpha=0.6, width=0.7)
        ax.set_xlabel("")
        ax.set_ylabel("Values of x")
        ax.set_xticks([])  # Rimuove i tick dell'asse X
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        if title:
            ax.set_title(f"{title} {len(frequency_map)} neurons")

        if limit is not None:
            ax.axhline(y=limit, color='r', linestyle='--', linewidth=1)

    # Rimuove eventuali subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, "multiple_frequency_graphs.png")
        plt.savefig(file_path)
        plt.close()

    if log_scale:
        plt.yscale("log")

    else:
        plt.show()



def plot_stacked_frequency_graphs(class_dict, limit=None, title=None, output_folder=None, log_scale=False):
    """
    Plotta un grafico a barre impilate (stacked bar chart) con più classi.

    Args:
        class_dict (dict): Dizionario con chiavi come classi e valori come liste di frequenze.
        limit (int, optional): Se specificato, disegna una linea orizzontale di soglia.
        title (str, optional): Titolo del grafico.
        output_folder (str, optional): Se specificato, salva l'immagine in questa cartella.
        log_scale (bool, optional): Se True, usa una scala logaritmica sull'asse Y.
    """
    class_names = list(class_dict.keys())  # Classi
    num_classes = len(class_names)  # Numero di classi
    x_labels = [i for i in range(class_dict[class_names[0]][0].shape[0])]

    # Assegna colori diversi per ogni classe
    class_colors = plt.cm.get_cmap('tab10', num_classes).colors

    # Convertiamo tutti i valori in array numpy per sicurezza
    class_data = {cls: np.asarray(vals).flatten() for cls, vals in class_dict.items()}

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros_like(x_labels, dtype=float)  # Base delle barre

    # Aggiunge le barre impilate
    for class_idx, (class_name, values) in enumerate(class_data.items()):
        ax.bar(x_labels, values, bottom=bottom, color=class_colors[class_idx], alpha=0.7, label=class_name)
        bottom += values  # Aggiorna la base per la prossima classe

    ax.set_xlabel("X Labels")
    ax.set_ylabel("Frequency")
    ax.set_title(title if title else "Stacked Bar Chart")
    ax.legend(title="Classes")

    # Se impostato, aggiunge una linea orizzontale di soglia
    if limit is not None:
        ax.axhline(y=limit, color='r', linestyle='--', linewidth=1, label="Limit")

    # Scala logaritmica se richiesto
    if log_scale:
        ax.set_yscale("log")

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Salva l'immagine se richiesto
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, "stacked_bar_chart.png")
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()


def get_data_report(class_unstable_dict, neg_stable_dict, pos_stable_dict):
    class_names = list(class_unstable_dict.keys())  # Classi
    num_classes = len(class_names)  # Numero di classi
    number_neurons = class_unstable_dict[class_names[0]][0].shape[0]

    list_of_unstable_tensor = list()
    list_of_neg_stable_tensor = list()
    list_of_pos_stable_tensor = list()

    for class_idx, (class_name, values) in enumerate(class_unstable_dict.items()):
        list_of_unstable_tensor.append(values[0])
        list_of_neg_stable_tensor.append(neg_stable_dict[class_name])
        list_of_pos_stable_tensor.append(pos_stable_dict[class_name])


    concatenated_tensor_unstable = torch.stack(list_of_unstable_tensor, dim=0).mean(dim=0)
    concatenated_tensor_neg_stable = torch.stack(list_of_neg_stable_tensor, dim=0).float().mean(dim=0)
    concatenated_tensor_pos_stable = torch.stack(list_of_pos_stable_tensor, dim=0).float().mean(dim=0)


    number_unstable_neurons = (concatenated_tensor_unstable!=0.0).sum()
    number_neg_stable_neurons = (concatenated_tensor_neg_stable==1.0).sum()
    number_pos_stable_neurons = (concatenated_tensor_pos_stable==1.0).sum()


    print(f"{number_unstable_neurons=}")
    print(f"{number_neg_stable_neurons=}")
    print(f"{number_pos_stable_neurons=}")