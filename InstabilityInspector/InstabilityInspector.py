import datetime
import os
import re
import random
import os
import glob

import pandas as pd
import torch
from onnx2keras import onnx_to_keras
from onnx import numpy_helper
from torch.utils.data import Subset, DataLoader

from InstabilityInspector.pynever_exe import py_run
from InstabilityInspector.utils import con2onnx, generate_lc_props, Bounds
import onnxruntime
import keras
import tensorflow as tf
import numpy as np
import onnx
import onnx2keras
from onnx2pytorch import ConvertModel


def generate_folders(*args):
    """
    This procedure create folders whose path is defined by the arguments
    :param args: the folders paths
    """

    for arg in args:
        # Check if the folder exists
        if os.path.exists(arg) and os.path.isdir(arg):
            # Look for all files in the specified folder
            all_files = glob.glob(os.path.join(arg, '*'))

            # Filter all non .txt files
            to_delete_file = [file for file in all_files if not file.endswith('.txt')]

            # Delete each found file
            for file in to_delete_file:
                try:
                    os.remove(file)
                except Exception as e:
                    raise ValueError(f"Impossible to remove file: {file}")
        else:
            os.makedirs(arg, exist_ok=True)


def dataset_cleaning(test_dataset):
    try:
        test_dataset = test_dataset.unbatch()
    except:
        print("Test dataset is not batched. Automatically converted")

    return test_dataset


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


class InstabilityInspector:
    def __init__(self, model_path, folder_path, test_dataset):

        # The neural network model must be in onnx format
        self.model_path = model_path

        # Load onnx model
        self.model = onnx.load(self.model_path)

        # Path where the folders will be created
        self.folder_path = folder_path

        # Test dataset in an unbatched form
        self.test_dataset = test_dataset

        # Paths for storing converted ONNX model and properties
        self.vnnlib_path = os.path.join(self.folder_path, "properties")
        self.output_path = os.path.join(self.folder_path, "output")

        # Path for storing generated data
        generate_folders(self.output_path, self.vnnlib_path)

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

    # def samples_inspector(self, number_of_samples: int, analysis_type: str):
    #     # Initialize lists to store sample images and labels
    #     samples_list = []
    #     labels_list = []
    #
    #     # Take the specified number of samples from the test dataset
    #     test_dataset = self.test_dataset.take(number_of_samples)
    #     for sample in test_dataset:
    #         # Reshape image to (1, 784) and convert to numpy array
    #         reshaped_image = tf.reshape(sample[0], (1, 784)).numpy()
    #         samples_list.append(reshaped_image)
    #
    #     # Initialize lists for bounds and other necessary variables
    #     bounds_list = []
    #
    #     # Get the number of neurons in each hidden layer
    #     number_of_neurons_per_layer = [self.weights_matrices[i].shape[1] for i in range(self.n_hidden_layers)]
    #
    #     # Initialize Bounds objects for each hidden layer
    #     for i in range(self.n_hidden_layers):
    #         i_layer_bounds = Bounds(number_of_neurons_per_layer[i])
    #         bounds_list.append(i_layer_bounds)
    #
    #     for index, sample in enumerate(samples_list):
    #         output = sample
    #         for i in range(self.n_layers):
    #             # Compute the output of each layer
    #             output = np.dot(output, self.weights_matrices[i]) + self.bias_matrices[i]
    #             if i != self.n_layers - 1:
    #                 # Update bounds for hidden layers
    #                 bounds_list[i].update_bounds(output.reshape(-1))
    #                 # Apply ReLU activation
    #                 output = np.maximum(0, output)
    #
    #     # Prepare data for CSV export
    #     write_dict = {}
    #     for x in range(self.n_hidden_layers):
    #         lower, upper = bounds_list[x].get_bounds()
    #         write_dict[f"lower_{x}"] = lower
    #         write_dict[f"upper_{x}"] = upper
    #
    #     # Create a DataFrame and export to CSV
    #     df = pd.DataFrame({k: pd.Series(v) for k, v in write_dict.items()})
    #     df.columns = self.labels_list
    #
    #     if analysis_type == "detailed":
    #         df.to_csv(
    #             os.path.join(self.samples_results_path, datetime.datetime.now().strftime("%Y%m%input-%H%M%S") + ".csv"),
    #             index=False)

    def bounds_inspector(self, number_of_samples: int, input_perturbation: float, output_perturbation: float,
                         complete: bool, analysis_type: str, output_file_name=None):
        """
        Inspects the bounds of the model using a specified number of samples and perturbations.

        :param number_of_samples: The number of samples for which the properties will be generated.
        :param input_perturbation: The perturbation in input for generating the properties.
        :param output_perturbation: The perturbation in output for generating the properties.
        :param complete: True if bounds are precise, otherwise they are over-approximated.
        :return: A list of dictionaries containing the bounds.
        """

        # Check that analysis type has a admitted value
        if not (analysis_type == "detailed" or analysis_type == "overall" or analysis_type == "both"):
            raise ValueError("analysis_type must be either 'detailed' or 'full'")

        # The analysis is run over an onnx model. In case of failure, the onnx model is converted into a Pytorch one
        pytorch_mode = False

        # This counts the number of samples wrongly classified
        violation_counter = 0

        try:
            # Open session to make inference on batch of the dataset
            session = onnxruntime.InferenceSession(self.model_path)

            # Get input and output names from the ONNX model
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

        except Exception as e:
            # Load ONNX model
            onnx_model = onnx.load(self.model_path)

            # Convert ONNX model to PyTorch
            pytorch_model = ConvertModel(onnx_model)
            pytorch_model.eval()  # Set the model to evaluation mode
            pytorch_mode = True

        # A restricted part of test set to generate the properties
        restricted_test_dataset = Subset(self.test_dataset, list(range(number_of_samples)))
        restricted_test_loader = DataLoader(restricted_test_dataset, batch_size=1, shuffle=False)

        io_pairs = []

        # Working with already flattened trained networks
        for batch_idx, (data, target) in enumerate(restricted_test_loader):
            # Ensure data and target are torch.Tensor objects
            if not isinstance(data, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("Expected data and target to be torch.Tensor objects")

            # Convert data to numpy array if needed
            data = data.numpy()
            target = target.numpy()

            # Transform dim in 2D for those models trained in batch
            if data.ndim == 1:
                data = data.reshape(1, -1)

            if pytorch_mode:
                # Perform inference using PyTorch model
                with torch.no_grad():
                    # Convert data back to tensor
                    data_tensor = torch.from_numpy(data).float()
                    output_tensor = pytorch_model(data_tensor)

                    # Convert output tensor to numpy
                    output = output_tensor.numpy()
            else:

                # Prepare input dictionary
                input_dict = {input_name: data}

                # Perform inference
                output = session.run([output_name], input_dict)

            # Convert output to flat list
            output_flat = output[0].flatten().tolist()

            # Convert data to flat list
            data_flat = data.flatten().tolist()

            # Store the predictions along with the target
            if np.argmax(output_flat) == target.flatten():
                io_pairs.append((data_flat, output_flat))
            else:
                violation_counter = violation_counter + 1

        if violation_counter / number_of_samples >= 0.8:
            #raise ValueError("Accuracy lower than 80%")
            pass

        # Properties are generated and stored in the specified path
        generate_lc_props(input_perturbation, output_perturbation, io_pairs, self.vnnlib_path)

        # Write a .txt report specifying the number of properties generated and the noises introduced
        self.write_properties_generation_report(number_of_samples, input_perturbation, output_perturbation)

        # Collection of dictionaries containing the bounds
        collected_dicts = []

        model_to_verify = os.path.join(self.folder_path, os.path.basename(self.model_path))

        for filename in os.listdir(self.vnnlib_path):
            if filename.endswith('.vnnlib'):
                i_property_path = os.path.join(self.vnnlib_path, filename)
                bounds_dict = py_run(str(model_to_verify), i_property_path, complete)
                bounds_dict.columns = self.labels_list
                collected_dicts.append(bounds_dict)

        if analysis_type == "detailed" or analysis_type == "both":
            self.write_csv(collected_dicts)

        if analysis_type == "overall" or analysis_type == "both":
            if output_file_name is not None:
                self.analyze(collected_dicts, output_file_name)
            else:
                self.analyze(collected_dicts)

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

        for index, file in enumerate(data):
            file_name = f"df_{index}" + ".csv"
            file_path = os.path.join(self.output_path, file_name)
            file.to_csv(file_path, index=False)

        report_path = os.path.join(self.output_path, 'report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write(f"Number of analysed properties: {len(data)}\n")
            for x in track_list:
                report_file.write(f"property: {x[1]}  bounds_file_name: {x[0]} \n")
        print(f"Report written to {report_path}")

    def analyze(self, collected_dicts, output_file_name="overall_analysis.csv"):
        results = list()
        for idx, df in enumerate(collected_dicts):
            unstable_neurons = list()

            for i in range(self.n_hidden_layers):
                lower_label = f"lower_{i}"
                upper_label = f"upper_{i}"

                bool_mask = (df[lower_label] < 0) & (df[upper_label] > 0)
                unstable_neurons.append(bool_mask.sum())

            temp_dict = {f"layer_{i}": unstable_neurons[i] for i in range(self.n_hidden_layers)}
            results.append(temp_dict)

        to_write_df = pd.DataFrame(results)
        to_write_df.to_csv(self.output_path + f"/{output_file_name}", index=False)
