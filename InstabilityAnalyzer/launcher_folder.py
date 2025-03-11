from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import Subset, DataLoader
from InstabilityAnalyzer.src.single_model_analysis import *
from InstabilityAnalyzer.src.folder_analysis import FolderModelAnalysis
# Percorso per salvare il dataset
folder_path = r"D:\3-9-25\rsloss\experiments\esperimento 15 - RSLoss con Batch\networks\mnist"
data_path =  "../datasets"
noise = 0.015
n_samples = 50

# Trasformazioni per normalizzare i dati
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Caricamento del dataset di training e test senza batching (batch_size=1)
test_dataset =FashionMNIST(root=data_path, train=False, download=True, transform=transform)
dataset_subset = Subset(dataset=test_dataset, indices=range(n_samples))
dataset_loader = DataLoader(dataset=dataset_subset, batch_size=1, shuffle=False)

analyzer = FolderModelAnalysis(folder_path, n_samples, dataset_loader)
analyzer.analyze_folder(noise)






