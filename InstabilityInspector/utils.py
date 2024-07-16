import keras
import onnxmltools
import os
import numpy as np
import re
from InstabilityInspector.pynever import datasets, nodes, networks
from InstabilityInspector.pynever.strategies import training, conversion
import torch
import torch.optim as opt
import os
import math


class Bounds:

    def __init__(self, dim: int):
        self.dim = dim
        self.lower_bounds = np.full(dim, np.inf)  # Limiti inferiori inizializzati a infinito positivo
        self.upper_bounds = np.full(dim, -np.inf)  # Limiti superiori inizializzati a infinito negativo

    def update_bounds(self, activation_values: np.ndarray):
        if activation_values.shape != self.lower_bounds.shape or activation_values.shape != self.upper_bounds.shape:
            raise ValueError("Le dimensioni degli array dei limiti non corrispondono.")

        self.lower_bounds = np.where(activation_values < self.lower_bounds, activation_values, self.lower_bounds)
        self.upper_bounds = np.where(activation_values > self.upper_bounds, activation_values, self.upper_bounds)

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def count_unstable_bounds(self):
        mask = (self.lower_bounds < 0) & (self.upper_bounds > 0)
        # Count the number of True values in the mask
        return np.count_nonzero(mask)

    def __repr__(self):
        return f"Bounds(lower_bounds={self.lower_bounds}, upper_bounds={self.upper_bounds})"


def con2onnx(model, onnx_folder_path: str):
    """ This function converts h5 keras model to ONNX model and save it """
    onnx_model = onnxmltools.convert_keras(model)
    os.makedirs(onnx_folder_path, exist_ok=True)
    onnxmltools.utils.save_model(onnx_model, onnx_folder_path + "/model.onnx")
    return onnx_model


def generate_lc_props(eps_noise: float, delta_tol: float, io_pairs: list, folder_path: str):
    # Property: x_i - eps_noise <= X_i <= x_i + eps_noise
    #           y_j - delta_tol <= Y_j <= y_j + delta_tol

    # generate folder for properties if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    i = 0
    for pair in io_pairs:
        if isinstance(pair[0], np.ndarray):
            pair_0 = pair[0].tolist()
        elif isinstance(pair[0], list):
            pair_0 = pair[0]
        else:
            raise ValueError("Input sample must be either numpy array or list.")

        n_inputs = len(pair_0)
        n_outputs = len(pair[1])

        with open(f'{folder_path}/loc_rob_property_{i}.vnnlib', 'w') as prop_file:
            for n in range(n_inputs):
                prop_file.write(f'(declare-const X_{n} Real)\n')
            prop_file.write('\n')

            for n in range(n_outputs):
                prop_file.write(f'(declare-const Y_{n} Real)\n')
            prop_file.write('\n')

            for n in range(n_inputs):
                prop_file.write(f'(assert (>= X_{n} {pair[0][n] - eps_noise}))\n')
                prop_file.write(f'(assert (<= X_{n} {pair[0][n] + eps_noise}))\n')
            prop_file.write('\n')

            for n in range(n_outputs):
                prop_file.write(f'(assert (>= Y_{n} {pair[1][n] - delta_tol}))\n')
                prop_file.write(f'(assert (<= Y_{n} {pair[1][n] + delta_tol}))\n')

        i += 1


def get_fc_weights_biases(model, verbose: bool = False):
    """
    Extract as numpy arrays the weights and biases matrices of the FC layers of the model in input in format onnx
    """

    # Initialize dictionaries to store weights and biases
    weights = []
    biases = []

    initializer = model.graph.initializer

    weights_pattern = re.compile(r'weight$')
    biases_pattern = re.compile(r'bias$')

    for i in initializer:
        if biases_pattern.search(i.name):
            biases.append(numpy_helper.to_array(i))
        elif weights_pattern.search(i.name):
            weights.append(numpy_helper.to_array(i))

    return weights, biases
