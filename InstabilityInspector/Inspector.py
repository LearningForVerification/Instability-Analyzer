import datetime
import os
import random
import pandas as pd
from onnx2keras import onnx_to_keras

from InstabilityInspector.pynever_exe import py_run
from InstabilityInspector.utils import con2onnx, generate_lc_props, Bounds
import keras
import tensorflow as tf
import numpy as np
import onnx
import onnx2keras

def generate_folders(*args):
    """
    This procedure create folders whose path is defined by the arguments
    :param args: the folders paths
    """
    for arg in args:
        os.makedirs(arg, exist_ok=True)


def dataset_cleaning(test_dataset):
    try:
        test_dataset = test_dataset.unbatch()
    except:
        print("Test dataset is not batched. Automatically converted")

    return test_dataset


def get_fc_weights_biases(model):
    """
    Extract as numpy arrays the weights and biases matrices of the FC layers of the model in input
    """

    weights_matrices = list()
    bias_matrices = list()

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, bias = layer.get_weights()
            weights_matrices.append(weights)
            bias_matrices.append(bias)

    return weights_matrices, bias_matrices


class Inspector:
    def __init__(self, model_path, folder_path, test_dataset):

        # The neural network model must be in .h5 format
        self.model_path = model_path

        # Load model
        if self.model_path.endswith('.h5'):
            self.model = keras.models.load_model(self.model_path)

        elif self.model_path.endswith('.onnx'):
            onnx_model = onnx.load(self.model_path)
            onnx.checker.check_model(onnx_model)
            self.model = onnx2keras.onnx_to_keras(onnx_model, ['X'])

        # Path where the folders will be created
        self.folder_path = folder_path

        # Test dataset in an unbatched form
        self.test_dataset = test_dataset

        # Paths for storing converted ONNX model and properties
        self.vnnlib_path = os.path.join(self.folder_path, "properties")
        self.bounds_results_path = os.path.join(self.folder_path, "bounds_results")
        self.samples_results_path = os.path.join(self.folder_path, "samples_results")

        # Path for storing generated data
        generate_folders(self.bounds_results_path, self.samples_results_path, self.vnnlib_path)

        # Clean test dataset
        self.test_dataset = dataset_cleaning(test_dataset)

        # Retrieve matrices and bias from model
        self.weights_matrices, self.bias_matrices = get_fc_weights_biases(self.model)

        # number of fc layers
        self.n_layers = len(self.weights_matrices)

        # number of hidden layers
        self.n_hidden_layers = self.n_layers - 1

        # Defining universal labels for pandas DataFrame
        self.labels_list = list()
        for i in range(self.n_hidden_layers):
            self.labels_list.append(f"lower_{i}")
            self.labels_list.append(f"upper_{i}")

    def samples_inspector(self, number_of_samples: int, to_write: bool):
        # Initialize lists to store sample images and labels
        samples_list = []
        labels_list = []

        # Take the specified number of samples from the test dataset
        test_dataset = self.test_dataset.take(number_of_samples)
        for sample in test_dataset:
            # Reshape image to (1, 784) and convert to numpy array
            reshaped_image = tf.reshape(sample[0], (1, 784)).numpy()
            samples_list.append(reshaped_image)

        # Initialize lists for bounds and other necessary variables
        bounds_list = []

        # Get the number of neurons in each hidden layer
        number_of_neurons_per_layer = [self.weights_matrices[i].shape[1] for i in range(self.n_hidden_layers)]

        # Initialize Bounds objects for each hidden layer
        for i in range(self.n_hidden_layers):
            i_layer_bounds = Bounds(number_of_neurons_per_layer[i])
            bounds_list.append(i_layer_bounds)

        for index, sample in enumerate(samples_list):
            output = sample
            for i in range(self.n_layers):
                # Compute the output of each layer
                output = np.dot(output, self.weights_matrices[i]) + self.bias_matrices[i]
                if i != self.n_layers - 1:
                    # Update bounds for hidden layers
                    bounds_list[i].update_bounds(output.reshape(-1))
                    # Apply ReLU activation
                    output = np.maximum(0, output)

        # Prepare data for CSV export
        write_dict = {}
        for x in range(self.n_hidden_layers):
            lower, upper = bounds_list[x].get_bounds()
            write_dict[f"lower_{x}"] = lower
            write_dict[f"upper_{x}"] = upper

        # Create a DataFrame and export to CSV
        df = pd.DataFrame({k: pd.Series(v) for k, v in write_dict.items()})
        df.columns = self.labels_list

        if to_write:
            df.to_csv(
                os.path.join(self.samples_results_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"),
                index=False)

    def bounds_inspector(self, number_of_samples: int, input_perturbation: float, output_perturbation: float,
                         complete: bool, to_write: bool):
        """
        Inspects the bounds of the model using a specified number of samples and perturbations.

        :param number_of_samples: The number of samples for which the properties will be generated.
        :param input_perturbation: The perturbation in input for generating the properties.
        :param output_perturbation: The perturbation in output for generating the properties.
        :param complete: True if bounds are precise, otherwise they are over-approximated.
        :param to_write: True if data must be written to disk, False otherwise.
        :return: A list of dictionaries containing the bounds.
        """

        restricted_test_dataset = self.test_dataset.take(number_of_samples)

        io_pairs = []

        if self.model_path.endswith('.h5'):
            model = keras.models.load_model(self.model_path)
            onnx_model = con2onnx(model, self.folder_path)

            # Extract the specified number of samples from the test dataset and generate their corresponding local
            # robustness properties
        else:
            onnx_model = onnx.load_model(self.model_path)
            k_model = onnx_to_keras(onnx_model, 'X')
            # Load the saved Keras model
            k_model_path = os.path.join(self.folder_path, "converted_model.h5")
            k_model.save(k_model_path)
            model = tf.keras.models.load_model(k_model_path)

        # Working with already flattened trained networks
        for sample in restricted_test_dataset:
            sample_x = list(sample[0].numpy())

            sample_y = model.predict(sample[0].numpy().reshape(1, -1))
            io_pairs.append((sample_x, list(sample_y.reshape(-1))))



        # Properties are generated and stored in the specified path
        generate_lc_props(input_perturbation, output_perturbation, io_pairs, self.vnnlib_path)

        # Write a .txt report specifying the number of properties generated and the noises introduced
        self.write_properties_generation_report(number_of_samples, input_perturbation, output_perturbation)

        # Collection of dictionaries containing the bounds
        collected_dicts = []

        model_to_verify = os.path.join(self.folder_path, "model.onnx")

        for filename in os.listdir(self.vnnlib_path):
            if filename.endswith('.vnnlib'):
                i_property_path = os.path.join(self.vnnlib_path, filename)
                bounds_dict = py_run(model_to_verify, i_property_path, complete)
                bounds_dict.columns = self.labels_list
                collected_dicts.append((bounds_dict, i_property_path))

        if to_write:
            self.write_csv(collected_dicts)

        return collected_dicts

    def write_properties_generation_report(self, number_of_samples, input_perturbation, output_perturbation):
        # Write a report specifying the number of properties generated and the perturbations used
        report_path = os.path.join(self.vnnlib_path, 'report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write(f"Number of properties generated: {number_of_samples}\n")
            report_file.write(f"Input perturbation: {input_perturbation}\n")
            report_file.write(f"Output perturbation: {output_perturbation}\n")
        print(f"Report written to {report_path}")

    def write_csv(self, data):
        """
        Takes a list of pandas DataFrames and writes them to CSV files.

        :param data: A list of pandas DataFrames
        :return: None
        """
        track_list = []

        for file in data:
            file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(
                random.randint(10000, 99999)) + ".csv"
            file_path = os.path.join(self.bounds_results_path, file_name)
            file[0].to_csv(file_path, index=False)
            track_list.append((file_name, file[1]))

        report_path = os.path.join(self.bounds_results_path, 'report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write(f"Number of analysed properties: {len(data)}\n")
            for x in track_list:
                report_file.write(f"property: {x[1]}  bounds_file_name: {x[0]} \n")
        print(f"Report written to {report_path}")
