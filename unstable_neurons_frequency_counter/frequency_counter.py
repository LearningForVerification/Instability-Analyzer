from InstabilityInspector.pynever.strategies.bp.bounds import HyperRectangleBounds
import os
import re

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
import InstabilityInspector.pynever.strategies.conversion as pyn_con
import random
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as tr
import torchvision



from InstabilityInspector.pynever_exe import py_run
from InstabilityInspector.utils import generate_lc_props

DEBUG = True
DATASET_DIR = "../dataset"



class FrequencyAnalyzer():
    def __init__(self, model_path, dataset):
        self.onnx_model = onnx.load(model_path)

        # Convert ONNX model to PyTorch
        self.model = ConvertModel(self.onnx_model)
        self.model.eval()  # Set the model to evaluation mode

        # Dataset, it works with MNIST, FMNIST, CIFAR
        self.dataset = dataset



    @staticmethod
    def extract_class_samples(dataset, class_dict):
        # Assuming the labels are stored in dataset.targets
        labels = dataset.targets.clone().detach()

        # List of subaset per class from class "0" to class "num_classes"
        subsets_dict = dict()


        for key, class_number in class_dict.items():
            # Get indices of samples that belong to the target class
            class_indices = torch.where(labels == int(key))[0]

            # Randomly sample `num_samples` indices from the class_indices
            sampled_indices = torch.randperm(len(class_indices))[:class_number]
            selected_samples = class_indices[sampled_indices]

            # Create a Subset dataset from the selected samples
            subset_dataset = Subset(dataset, selected_samples)
            subsets_dict[int(key)] = subset_dataset

            if DEBUG:
                for key_subset, subset in subsets_dict.items():
                    print(f"Sample number at key {key_subset} : {len(subset)}")

        return subsets_dict


    def get_frequency_multi_class(self, class_dict, eps_noise, delta_tol):
        subsets_dict = FrequencyAnalyzer.extract_class_samples(self.dataset, class_dict)

        multi_class_dict = dict()

        keys = list(subsets_dict.keys())

        for key, subset in subsets_dict.items():
            # contains a list whose elements are numpy arrays representing the unstable nodes per neuron
            i_class_list = self.get_frequency_single_class(subset, eps_noise, delta_tol)
            multi_class_dict[key] = i_class_list

        return multi_class_dict




    def get_frequency_single_class(self, i_subset, eps_noise, delta_tol) -> list:
        io_pairs = list()

        subset_loader = DataLoader(i_subset, batch_size=1, shuffle=False)

        # Working with already flattened trained networks
        for batch_idx, (data, target) in enumerate(subset_loader):
            # Ensure data and target are torch.Tensor objects
            if not isinstance(data, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("Expected data and target to be torch.Tensor objects")

            # Convert data to a numpy array if needed
            data = data.numpy()
            target = target.numpy()

            # Transform dim in 2D for those models trained in batch
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # Perform inference using PyTorch model
            with torch.no_grad():
                # Convert data back to tensor
                data_tensor = torch.from_numpy(data).float()
                output_tensor = self.model(data_tensor)

                # Convert output tensor to numpy
                output = output_tensor.numpy()

            # Convert output to flat list
            output_flat = output[0].flatten()

            # Convert data to flat list
            data_flat = data.flatten()

            # Store the predictions along with the target
            if np.argmax(output_flat) == target.flatten():
                io_pairs.append((data_flat, output_flat))
        #    else:
        #        violation_counter = violation_counter + 1

        #if violation_counter / len(subset_loader) >= 0.8:
            #raise ValueError("Accuracy lower than 80%")

        # Given the io_pair list, it necessary to generate the HyperRectangles object
        bounds_object_list = list()

        for pair in io_pairs:
            input = pair[0]
            output = pair[1]

            lower = input - eps_noise
            upper = input + eps_noise

            rect = HyperRectangleBounds(lower, upper)
            bounds_object_list.append(rect)


        # Running a bound propagation algorithm for each bound rectangle
        df_bounds_list = list()
        for rect in bounds_object_list:
            net_id = ''.join(str(random.randint(0, 9)) for _ in range(5))

            onnx_network = pyn_con.ONNXNetwork(net_id, self.onnx_model)
            network = pyn_con.ONNXConverter().to_neural_network(onnx_network)

            bounds_manager = bp.BoundsManager(network, None)
            overapprox_df_dict = bounds_manager.return_df_dict(converted_input=rect)
            df_bounds_list.append(overapprox_df_dict)

        # Retrieve the number of columns
        num_hidden_layers = len(df_bounds_list[0].columns)//2

        # Results
        unstable_node_counts_per_layer = []  # Renamed for clarity

        # Iterate through each layer
        for i in range(num_hidden_layers):

            # List to hold boolean masks for each dataframe
            bool_mask_list = []

            for df in df_bounds_list:
                # Create a boolean mask based on the conditions
                bool_mask = (df[df.columns[i * 2]] < 0) & (df[df.columns[i * 2 + 1]] > 0)

                # Convert boolean mask to NumPy array and append it to the list
                bool_mask_list.append(bool_mask.to_numpy())

            # Stack the boolean masks as columns in a single NumPy array
            result_array = np.column_stack(bool_mask_list)

            # Sum the boolean values along the rows to count unstable nodes per layer
            unstable_node_counts_per_layer.append(result_array.sum(axis=1))

        return unstable_node_counts_per_layer



if __name__ == '__main__':
    model_path = r"C:\Users\andr3\Desktop\MNIST\no_batch incredible results\1100.onnx"

    # Data loading and transformation for the MNIST dataset
    transform = tr.Compose([
        tr.ToTensor(),  # Convert image to tensor
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image to a 1D tensor
    ])

    # Load the MNIST training dataset (for consistency, though not used in this script)
    train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)

    # Load the MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

    frequency_analyzer = FrequencyAnalyzer(model_path, train_dataset)
    class_dict = {
        "0": 50,
        "1": 50,
        "2": 50,
        "3": 50,
        "4": 50,
        "5": 50,
        "6": 50,
        "7": 50,
        "8": 50,
        "9": 50
    }

    multiclass_class_dict = frequency_analyzer.get_frequency_multi_class(class_dict, 0.015, 0.03)

    #TODO
    for key, value in multiclass_class_dict.items():
        multiclass_class_dict[key] = value[0]

    df = pd.DataFrame(multiclass_class_dict)
    df.to_csv("output.csv", index=False)
    pass















