import argparse
import os

import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

DATASET_DIR = "dataset"


def analyze_folder(networks_folder_path: str, results_folder_path: str, number_of_samples: int, input_perturbation: float, output_perturbation: float,
                   complete: bool, analysis_type: str, check_accuracy: bool):

    # folder path where the onnx model are
    for index, file in enumerate(os.listdir(networks_folder_path)):
        file = os.path.join(networks_folder_path, file)
        # check if the file has onnx extension
        if os.path.isfile(file) and file.endswith(".onnx"):
            file_name = f"overall_analysis_{index}"

            single_nn_inspector = InstabilityInspector(file, results_folder_path, test_dataset)
            dict = single_nn_inspector.bounds_inspector(number_of_samples, input_perturbation, output_perturbation, complete, analysis_type, check_accuracy, output_file_name=file_name)




if __name__ == '__main__':
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Analyze neural network performance with MNIST dataset.')

    parser.add_argument('--results_folder_path', type=str, default="experiments",
                        help='Path to save the analysis results.')
    parser.add_argument('--networks_folder_path', type=str, required=True,
                        help='Path to the folder containing neural networks.')
    parser.add_argument('--number_of_samples', type=int, default=5,
                        help='Number of samples to use for analysis.')
    parser.add_argument('--input_perturbation', type=float, default=0.05,
                        help='Magnitude of input perturbation for analysis.')
    parser.add_argument('--output_perturbation', type=float, default=0.15,
                        help='Magnitude of output perturbation for analysis.')
    parser.add_argument('--complete', action='store_true',
                        help='Perform a complete analysis if this flag is set.')
    parser.add_argument('--analysis_type', type=str, default="overall", choices=["overall", "detailed", "both"],
                        help='Type of analysis to perform.')
    parser.add_argument('--check_accuracy', type=bool, default=True,
                        help='Check accuracy of the samples given in input.')

    args = parser.parse_args()

    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])

    train_dataset = torchvision.datasets.MNIST('DATASET_DIR', train=True, download=True, transform=transform)

    global test_dataset
    test_dataset = torchvision.datasets.MNIST('DATASET_DIR', train=False, download=True, transform=transform)

    # Call the analyze_folder function with the parsed arguments
    analyze_folder(
        networks_folder_path=args.networks_folder_path,
        results_folder_path=args.results_folder_path,
        number_of_samples=args.number_of_samples,
        input_perturbation=args.input_perturbation,
        output_perturbation=args.output_perturbation,
        complete=args.complete,
        analysis_type=args.analysis_type,
        check_accuracy=args.check_accuracy
    )

    #python your_script.py --networks_folder_path "C:\path\to\NNs_folder" --results_folder_path "experiments"
    # --number_of_samples 10 --input_perturbation 0.1 --output_perturbation 0.2 --complete --analysis_type "detailed" --check_accuracy True
