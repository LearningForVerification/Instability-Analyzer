import random
import re
import os
import re
import InstabilityInspector.pynever.strategies.conversion as pyn_con


import numpy as np
import onnx
import onnxruntime
import pandas as pd
import torch
import glob
from onnx import numpy_helper
from onnx2pytorch import ConvertModel
from torch.utils.data import Subset, DataLoader

import InstabilityInspector.pynever.strategies.bp.bounds_manager as bp
from InstabilityInspector.pynever_exe import py_run
from InstabilityInspector.utils import generate_lc_props, hyperect_properties
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
REGRESSION = True


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
        self.output_path = os.path.join(self.folder_path, "output")

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


    def bounds_inspector(self, number_of_samples: int, input_perturbation: float, complete: bool, analysis_type: str,
                         check_accuracy: bool = True, output_file_name=None):
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
            output_flat = output[0].flatten()

            # Convert data to flat list
            data_flat = data.flatten()


            if not REGRESSION:
                # Store the predictions along with the target
                if np.argmax(output_flat) == target.flatten():
                    io_pairs.append((data_flat, output_flat))
                else:
                    violation_counter = violation_counter + 1
            else :
                io_pairs.append((data_flat, output_flat))

        if violation_counter / number_of_samples >= 0.2 and check_accuracy:
            raise ValueError("Accuracy lower than 80%")

        properties_list = hyperect_properties(input_perturbation, io_pairs)

        # Collection of dictionaries containing the bounds
        collected_dicts = []

        for property in properties_list:
            net_id = ''.join(str(random.randint(0, 9)) for _ in range(5))

            onnx_network = pyn_con.ONNXNetwork(net_id, self.model)
            network = pyn_con.ONNXConverter().to_neural_network(onnx_network)

            bounds_manager = bp.BoundsManager(network, None)
            overapprox_df_dict = bounds_manager.return_df_dict(converted_input=property)
            collected_dicts.append(overapprox_df_dict)

        if analysis_type == "detailed" or analysis_type == "both":
            self.write_csv(collected_dicts)

        if analysis_type == "overall" or analysis_type == "both":
            if output_file_name is not None:
                overall_dict = self.analyze(collected_dicts, output_file_name)
            else:
                overall_dict = self.analyze(collected_dicts)

        return collected_dicts, overall_dict

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
                # Extract the two columns corresponding to the lower and upper bounds
                lower_column = df.columns[i * 2]
                upper_column = df.columns[i * 2 + 1]

                # Create a boolean mask for unstable neurons (lower < 0 and upper > 0)
                bool_mask = (df[lower_column] < 0) & (df[upper_column] > 0)

                # Count the number of unstable neurons and append to the list
                unstable_neurons.append(bool_mask.sum())

            # Create a dictionary for the current DataFrame results
            temp_dict = {f"layer_{i}": unstable_neurons[i] for i in range(self.n_hidden_layers)}
            results.append(temp_dict)

        df_results = pd.DataFrame(results)
        # Save results to CSV (this part is missing in the provided code but can be added)
        return df_results

