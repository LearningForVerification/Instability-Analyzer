from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader
from InstabilityAnalyzer.src.single_model_analysis import *
# Percorso per salvare il dataset
model_path = r"C:\Users\andr3\PycharmProjects\Instability-Analyzer\InstabilityAnalyzer\256-128-64-32.onnx"
data_path =  "../datasets"
n_samples = 10
noise = 0.015


# Trasformazioni per normalizzare i dati
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Caricamento del dataset di training e test senza batching (batch_size=1)
test_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)
dataset_subset = Subset(dataset=test_dataset, indices=range(n_samples))
dataset_loader = DataLoader(dataset=dataset_subset, batch_size=1, shuffle=False)

analyzer = AnalyzeModel(model_path, dataset_loader, n_samples=n_samples)
bounds = analyzer.compute_bounds(noise=noise)
results = analyzer.analyze(bounds).get_average()
frequency_map, _, _ = analyzer.analyze_neurons_activation_frequency(bounds)
plot_frequency_graphs(frequency_map, output_folder)
pass






