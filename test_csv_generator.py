import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr

DATASET = "fashion_mnist"
DATASET_DIR = "dataset"
MEAN = (0.1307,)
STD = (0.3081,)

if __name__ == '__main__':
    # Data loading and transformations
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Normalize(MEAN, STD),
        tr.Lambda(lambda x: torch.flatten(x))  # Flatten the image
    ])

    train_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(DATASET_DIR, train=False, download=True, transform=transform)

    inspector = InstabilityInspector(r"/Users/andrea/Desktop/Instability-Analizer/experiments/model1.onnx", "experiments",
                                     test_dataset)
    dict = inspector.bounds_inspector(5, 0.05, 0.15, False, "overall")


