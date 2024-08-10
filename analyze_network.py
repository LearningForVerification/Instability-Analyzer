import argparse

import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

DATASET_DIR = "dataset"

if __name__ == '__main__':
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Instability analysis of neural networks with MNIST dataset.')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the ONNX model file.')
    parser.add_argument('--results_folder_path', type=str, default="experiments",
                        help='Path to save the analysis results.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory where the MNIST dataset will be stored/downloaded.')
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
    parser.add_argument('--check_accuracy', type=bool, default=True, help='Check accuracy .')

    args = parser.parse_args()

    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])

    train_dataset = torchvision.datasets.MNIST(args.dataset_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(args.dataset_dir, train=False, download=True, transform=transform)

    # Create an instance of InstabilityInspector with the parsed arguments
    inspector = InstabilityInspector(args.model_path, args.results_folder_path, test_dataset)

    # Perform bounds inspection
    result_dict = inspector.bounds_inspector(
        args.number_of_samples,
        args.input_perturbation,
        args.output_perturbation,
        args.complete,
        args.analysis_type,
        args.check_accuracy

    )

    # python your_script.py --model_path "/Users/andrea/Desktop/Instability-Analizer/experiments/model1.onnx"
    # --results_folder_path "experiments" --dataset_dir "DATASET_DIR" --number_of_samples 10 --input_perturbation 0.1 --output_perturbation 0.2 --complete --analysis_type "detailed" --check_accuracy True