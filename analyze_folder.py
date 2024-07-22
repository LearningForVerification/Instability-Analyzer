import os

import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

DATASET_DIR = "dataset"

if __name__ == '__main__':
    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])

    train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

    inspector = InstabilityInspector(r"/Users/andrea/Desktop/Instability-Analizer/experiments/model1.onnx",
                                     "experiments",
                                     test_dataset)
    dict = inspector.bounds_inspector(5, 0.05, 0.15, False, "overall")


def analyze_folder(networks_folder_path: str, results_folder_path: str, number_of_samples: int, input_perturbation: float, output_perturbation: float,
                   complete: bool, analysis_type: str):

    # folder path where the onnx model are
    for index, file in enumerate(os.listdir(networks_folder_path)):
        # check if the file has onnx extension
        if os.path.isfile(file) and file.endswith(".onnx"):
            file_name = f"overall_analysis_{index}"

            single_nn_inspector = InstabilityInspector(file, results_folder_path, test_dataset)
            dict = inspector.bounds_inspector(5, 0.05, 0.15, False, "overall", output_file_name=file_name)

