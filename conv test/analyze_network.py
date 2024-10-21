import argparse
import torch
import torchvision
from InstabilityInspector.InstabilityInspector import InstabilityInspector
import torchvision.transforms as tr
import pickle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.onnx


class DatasetFromCSV(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Ottieni la riga dal DataFrame
        item = self.dataframe.iloc[idx]

        # Converti la stringa di input in un array NumPy
        input = item["input"]

        # Converti l'array NumPy in un tensore PyTorch
        input_tensor = torch.from_numpy(input).float().reshape(1, 4, 4)


        # Converti anche il label (target) in un tensore PyTorch
        target = torch.tensor(item["label"]).float()

        return input_tensor, target




if __name__ == '__main__':
    # DATASET FOR SMALL NETOWORK
    # test_df = pd.read_pickle(r"C:\Users\andr3\Desktop\Instability-Analizer\conv test\synth_dataset\test_dataset.pkl")
    #
    # # Creare il DataLoader per il training e il test
    # test_dataset = DatasetFromCSV(test_df)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Preprocessing del dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converti le immagini in tensori
    ])

    # Carica il dataset MNIST
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Estrai i dati e le etichette
    X = mnist_data.data.numpy()  # Dati delle immagini
    y = mnist_data.targets.numpy()  # Etichette

    # Reshape delle immagini in formato (batch_size, 1, 28, 28)
    XConv = X.reshape(-1, 1, 28, 28)

    # Dividi il dataset in training e validation set
    X_train, X_val, y_train, y_val = train_test_split(XConv, y, test_size=0.1, random_state=42)

    # Converti i dati in tensori PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Crea DataLoader per PyTorch
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create an instance of InstabilityInspector with the parsed arguments
    inspector = InstabilityInspector(r"C:\Users\andr3\Desktop\Instability-Analizer\conv test\mnist_model.onnx", "results", val_dataset)

    # Perform bounds inspection using the provided parameters
    result_dict = inspector.bounds_inspector(1, 0.015, False, "overall", False)
    pass