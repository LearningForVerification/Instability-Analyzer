import argparse
import os
import torch
import torchvision
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as transforms

DATASET_DIR = "dataset"


# Function to analyze all ONNX models in a specified folder
def analyze_folder(networks_folder_path: str, results_folder_path: str, number_of_samples: int,
                   input_perturbation: float, output_perturbation: float, complete: bool,
                   analysis_type: str, check_accuracy: bool, test_dataset):
    """
    Analyzes all ONNX models in the specified folder using the InstabilityInspector tool.

    Parameters:
    - networks_folder_path (str): Path to the folder containing ONNX models.
    - results_folder_path (str): Directory where analysis results will be saved.
    - number_of_samples (int): Number of samples to use for analysis.
    - input_perturbation (float): Magnitude of input perturbation to apply.
    - output_perturbation (float): Magnitude of output perturbation to apply.
    - complete (bool): Whether to perform a complete analysis.
    - analysis_type (str): Type of analysis to perform ('overall', 'detailed', 'both').
    - check_accuracy (bool): Flag to check the accuracy of the samples.
    - test_dataset (Dataset): The dataset to be used for testing the models.
    """

    # Iterate through all files in the networks folder
    for index, file_name in enumerate(os.listdir(networks_folder_path)):
        file_path = os.path.join(networks_folder_path, file_name)

        # Check if the file is an ONNX model
        if os.path.isfile(file_path) and file_name.endswith(".onnx"):
            # Generate a unique filename for the analysis results
            analysis_filename = f"overall_analysis_{file_name}"
            analysis_filename.replace(".onnx", "")

            # Initialize the InstabilityInspector for the current ONNX model
            inspector = InstabilityInspector(file_path, results_folder_path, test_dataset)

            # Perform the bounds analysis on the model
            inspector.bounds_inspector(number_of_samples, input_perturbation, complete, analysis_type, check_accuracy,
                                       output_file_name=analysis_filename)


if __name__ == '__main__':
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description='Analyze neural network performance using the MNIST dataset.')

    # Argument for specifying the results folder path
    parser.add_argument('--results_folder_path', type=str, default="experiments",
                        help='Path to save the analysis results.')

    # Argument for specifying the networks folder path
    parser.add_argument('--networks_folder_path', type=str, required=True,
                        help='Path to the folder containing neural network models.')

    # Argument for specifying the number of samples to use in the analysis
    parser.add_argument('--number_of_samples', type=int, default=5,
                        help='Number of samples to use for the analysis.')

    # Argument for specifying the magnitude of input perturbation
    parser.add_argument('--input_perturbation', type=float, default=0.05,
                        help='Magnitude of input perturbation for the analysis.')

    # Argument for specifying the magnitude of output perturbation
    parser.add_argument('--output_perturbation', type=float, default=0.15,
                        help='Magnitude of output perturbation for the analysis.')

    # Flag for performing a complete analysis
    parser.add_argument('--complete', action='store_true',
                        help='Perform a complete analysis if this flag is set.')

    # Argument for specifying the type of analysis to perform
    parser.add_argument('--analysis_type', type=str, default="overall", choices=["overall", "detailed", "both"],
                        help='Type of analysis to perform.')

    # Argument for checking the accuracy of the samples
    parser.add_argument('--check_accuracy', action='store_true',
                        help='Flag to check the accuracy of the samples given as input.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Data loading and transformation for the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the image to a 1D tensor
    ])

    # Load the MNIST training dataset (for consistency, though not used in this script)
    train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)

    # Load the MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

    # Call the analyze_folder function with the parsed arguments
    analyze_folder(
        networks_folder_path=args.networks_folder_path,
        results_folder_path=args.results_folder_path,
        number_of_samples=args.number_of_samples,
        input_perturbation=args.input_perturbation,
        output_perturbation=args.output_perturbation,
        complete=args.complete,
        analysis_type=args.analysis_type,
        check_accuracy=args.check_accuracy,
        test_dataset=test_dataset
    )
