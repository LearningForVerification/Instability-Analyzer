
import argparse
import torch
import torchvision
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

from analyze_folder import analyze_folder

# Directory where the MNIST dataset will be stored/downloaded
DATASET_DIR = "dataset"
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':
    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])

    train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)

    global test_dataset
    test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

    results_folder_path = "experiments"
    networks_folder_path = r"C:\Users\andr3\Desktop\Instability-Analizer\NNs_folder"
    analyze_folder(networks_folder_path=networks_folder_path, results_folder_path=results_folder_path, number_of_samples=5, input_perturbation=0.05, output_perturbation=0.15, complete=False, analysis_type="overall", check_accuracy=True, test_dataset=test_dataset)