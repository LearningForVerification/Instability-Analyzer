import argparse
import torch
import torchvision
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

# Directory where the MNIST dataset will be stored/downloaded
DATASET_DIR = "dataset"

if __name__ == '__main__':
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Instability analysis of neural networks with MNIST dataset.')

    # Argument for specifying the path to the ONNX model file
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the ONNX model file.')

    # Argument for specifying the results folder path
    parser.add_argument('--results_folder_path', type=str, default="experiments",
                        help='Path to save the analysis results.')

    # Argument for specifying the dataset directory
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory where the MNIST dataset will be stored/downloaded.')

    # Argument for specifying the number of samples to use in the analysis
    parser.add_argument('--number_of_samples', type=int, default=5,
                        help='Number of samples to use for analysis.')

    # Argument for specifying the magnitude of input perturbation
    parser.add_argument('--input_perturbation', type=float, default=0.05,
                        help='Magnitude of input perturbation for analysis.')

    # Argument for specifying the magnitude of output perturbation
    parser.add_argument('--output_perturbation', type=float, default=0.15,
                        help='Magnitude of output perturbation for analysis.')

    # Flag for performing a complete analysis
    parser.add_argument('--complete', type=bool, default=False,
                        help='Perform a complete analysis if this flag is set.')

    # Argument for specifying the type of analysis to perform
    parser.add_argument('--analysis_type', type=str, default="overall", choices=["overall", "detailed", "both"],
                        help='Type of analysis to perform.')

    # Argument for checking the accuracy of the samples
    parser.add_argument('--check_accuracy', type=bool, default=True, help='Check the accuracy during the analysis.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),  # Convert image to tensor
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image into a 1D tensor
    ])

    # Load the MNIST training dataset (for completeness, though not used directly)
    train_dataset = torchvision.datasets.MNIST(args.dataset_dir, train=True, download=True, transform=transform)

    # Load the MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(args.dataset_dir, train=False, download=True, transform=transform)

    # Create an instance of InstabilityInspector with the parsed arguments
    inspector = InstabilityInspector(args.model_path, args.results_folder_path, test_dataset)

    # Perform bounds inspection using the provided parameters
    result_dict = inspector.bounds_inspector(
        args.number_of_samples,
        args.input_perturbation,
        args.output_perturbation,
        args.complete,
        args.analysis_type,
        args.check_accuracy
    )
