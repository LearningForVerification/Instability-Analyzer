from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset, DataLoader
from InstabilityAnalyzer.src.single_model_analysis import *
from pynever import datasets
import torch

# Percorso per il modello e il dataset
model_path = r"C:\Users\andr3\Desktop\Risultati Paper\esperimento 1 - Adam vs SGD\mnist\Adam\one_layered_networks\200.onnx"
data_path = "../datasets"
n_samples_per_class = 100  # Numero di campioni per classe
noise = 0.015  # Rumore da applicare


def get_dataset_of_class_i(dataset, selected_classes: list):
    """
    Filtra il dataset per selezionare solo le classi desiderate.

    :param dataset: Dataset completo
    :param selected_classes: Lista delle classi da selezionare
    :return: Sottoinsieme del dataset contenente solo le classi selezionate
    """
    filtered_data = [(img, label) for img, label in zip(dataset.data, dataset.targets) if label in selected_classes]
    filtered_images, filtered_labels = zip(*filtered_data)
    return torch.utils.data.TensorDataset(torch.stack(filtered_images), torch.tensor(filtered_labels))


# Definizione delle trasformazioni per normalizzare i dati
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Caricamento del dataset MNIST
dataset = MNIST(root=data_path, train=True, download=True, transform=transform)

# Dizionari per memorizzare i risultati delle analisi
class_dict_unstable = {}
class_dict_pos_stable = {}
class_dict_neg_stable = {}

selected_classes = list(range(10))  # Classi da 0 a 9

for label in selected_classes:
    print(f"Analisi della classe {label}")

    # Selezione dei dati relativi alla classe specificata
    dataset_reduced = get_dataset_of_class_i(dataset, selected_classes=[label])
    dataset_subset = Subset(dataset=dataset_reduced, indices=range(n_samples_per_class))
    dataset_loader = DataLoader(dataset=dataset_subset, batch_size=1, shuffle=False)

    # Analisi del modello con i dati della classe
    analyzer = AnalyzeModel(model_path, dataset_loader, n_samples=n_samples_per_class)
    bounds = analyzer.compute_bounds(noise=noise)
    results = analyzer.analyze(bounds).get_average()
    print(f"Risultati per la classe {label}: {results}")

    # Analisi delle attivazioni dei neuroni
    un_frequency_map, stable_pos_frequency_map, stable_neg_frequency_map = analyzer.analyze_neurons_activation_frequency(
        bounds)

    class_dict_unstable[str(label)] = un_frequency_map
    stable_negative_map = stable_neg_frequency_map[0] == n_samples_per_class
    stable_positive_map = stable_pos_frequency_map[0] == n_samples_per_class

    class_dict_neg_stable[str(label)] = stable_negative_map
    class_dict_pos_stable[str(label)] = stable_positive_map

# Generazione del grafico delle frequenze instabili
plot_stacked_frequency_graphs(
    class_dict_unstable,
    limit=n_samples_per_class * len(selected_classes),
    title="Stacked Bar Chart on All_classes MNIST Rete Shallow",
    log_scale=True
)

# Creazione del report dei dati
#unstable = get_data_report(class_dict_unstable, class_dict_neg_stable, class_dict_pos_stable)
