import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.onnx

# Funzione di training
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Azzera i gradienti
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Custom neural network class
class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.conv2_dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(4608, 100)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_dropout(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)  # Flattening the tensor
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# Funzione di valutazione sul validation set
def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Set device for computation (cuda if available, otherwise cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model, move it to the appropriate device
model = CustomNN().to(device)

# Imposta la funzione di perdita (loss) e l'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# Addestra il modello
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Valuta il modello
evaluate_model(model, val_loader)

# Definisci un input fittizio con la stessa forma del batch (ad esempio, batch_size=1, channels=1, height=28, width=28)
dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Esporta il modello in formato ONNX
torch.onnx.export(
    model,                          # Il modello PyTorch
    dummy_input,                    # L'input fittizio
    "mnist_model.onnx",             # Il nome del file ONNX
    input_names=["input"],           # Nome dell'input
    output_names=["output"],         # Nome dell'output
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},)

