import torch
from torch.utils.data import DataLoader
import os
import csv
from single_model_analysis import AnalyzeModel

class FolderModelAnalysis:
    def __init__(self, folder_path, n_samples, dataset_loader):
        self.folder_path = folder_path
        self.n_samples = n_samples
        self.dataset_loader = dataset_loader



    def analyze_folder(self, noise, output_path=None):
        """ Analizza i file .onnx e salva i risultati in un CSV con labels come intestazioni di colonna. """

        # Verifica che la cartella esista
        if not os.path.isdir(self.folder_path):
            print(f"Errore: La cartella '{self.folder_path}' non esiste.")
            return

        # Filtra solo i file con estensione .onnx
        onnx_files = [f for f in os.listdir(self.folder_path) if f.endswith('.onnx')]

        if not onnx_files:
            print("Nessun file .onnx trovato nella cartella.")
            return

        csv_dict = {}

        for file in onnx_files:
            file_path = os.path.join(self.folder_path, file)

            # Analizza il modello ONNX
            analyzer = AnalyzeModel(file_path, self.dataset_loader, self.n_samples)
            bounds = analyzer.compute_bounds(noise= noise)
            result = analyzer.analyze(bounds).get_average()

            csv_dict[int(file.replace(".onnx", ""))] = result

        # Ordering dict per key
        csv_dict = {k: csv_dict[k] for k in sorted(csv_dict)}

        # Definisce il percorso del file CSV
        output_folder = output_path if output_path else os.getcwd()
        os.makedirs(output_folder, exist_ok=True)
        csv_file = os.path.join(output_folder, "analysis_results.csv")

        # Scrive i dati nel file CSV
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)

            # Scrive l'intestazione (le chiavi del dizionario)
            writer.writerow(csv_dict.keys())
            writer.writerow(csv_dict.values())

        print(f"Analisi completata. Risultati salvati in: {csv_file}")
