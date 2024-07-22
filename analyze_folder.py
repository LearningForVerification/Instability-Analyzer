import os

import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

DATASET_DIR = "dataset"


def analyze_folder(networks_folder_path: str, results_folder_path: str, number_of_samples: int, input_perturbation: float, output_perturbation: float,
                   complete: bool, analysis_type: str):

    # folder path where the onnx model are
    for index, file in enumerate(os.listdir(networks_folder_path)):
        file = os.path.join(networks_folder_path, file)
        # check if the file has onnx extension
        if os.path.isfile(file) and file.endswith(".onnx"):
            file_name = f"overall_analysis_{index}"

            single_nn_inspector = InstabilityInspector(file, results_folder_path, test_dataset)
            dict = single_nn_inspector.bounds_inspector(number_of_samples, input_perturbation, output_perturbation, complete, analysis_type, output_file_name=file_name)



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
    networks_folder_path = r"C:\Users\Andrea\Desktop\Instability-Analizer\NNs_folder"
    analyze_folder(networks_folder_path=networks_folder_path, results_folder_path=results_folder_path, number_of_samples=5, input_perturbation=0.05, output_perturbation=0.15, complete=False, analysis_type="overall")


