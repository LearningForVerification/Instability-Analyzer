import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Dataset
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

# Definizione del modello
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Primo livello convoluzionale: 8 filtri 2x2, senza padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, padding=0)

        self.flatten = nn.Flatten()

        # Livello Fully Connected (dopo Flatten)
        self.fc1 = nn.Linear(8 * 3 * 3, 10)  # 8 feature map 3x3 dopo convoluzione
        self.fc2 = nn.Linear(10, 1)         # Output a singolo nodo per regressione

    def forward(self, x):
        # Passa attraverso la convoluzione e poi applica l'attivazione ReLU
        x = self.conv1(x)
        x = F.relu(x)

        # Appiattisce i dati (flatten)
        x = self.flatten(x)

        # Livelli fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Funzione di training
def train(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Azzerare i gradienti
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # Aggiungi dimensione per il target

            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            # Sommare la perdita (loss)
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Funzione di valutazione sul test set
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # Aggiungi dimensione per il target
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

from torch import optim

if __name__ == '__main__':

    # Istanziare il modello
    model = ConvNet()

    # Impostare la loss function e l'ottimizzatore
    criterion = nn.MSELoss()  # Per regressione usiamo la Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Caricare i dati di allenamento e test
    train_df = pd.read_pickle(r"C:\Users\andr3\Desktop\Instability-Analizer\conv test\synth_dataset\train_dataset.pkl")
    test_df = pd.read_pickle(r"C:\Users\andr3\Desktop\Instability-Analizer\conv test\synth_dataset\test_dataset.pkl")

    # Creare il DataLoader per il training e il test
    train_dataset = DatasetFromCSV(train_df)
    test_dataset = DatasetFromCSV(test_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Allenare il modello
    train(model, train_loader, criterion, optimizer, epochs=10)

    # Valutare il modello sui dati di test
    evaluate(model, test_loader, criterion)

    # Esportare il modello in formato ONNX
    dummy_input = torch.randn(1, 1, 4, 4)  # Input fittizio per l'esportazione
    torch.onnx.export(model, dummy_input, "model.onnx",
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    print("Modello esportato in formato ONNX.")
    torch.save(model, 'model.pth')
