import os
import pandas as pd


class InstabilityAnalysis:
    def __init__(self, input_folder, output_folder):
        self.input_path = input_folder
        self.output_path = output_folder

    def analyze(self):
        # create a list containing all files in the folder
        all_files = os.listdir(self.input_path)
        file_csv = [file for file in all_files if file.endswith('.csv')]

        csv_file_paths = [os.path.join(self.input_path, file) for file in file_csv]
        output_file_path = os.path.join(self.output_path, 'instability_analysis.csv')
        results = list()

        for file in csv_file_paths:
            df = pd.read_csv(file)
            layer_numbers = int(df.columns.shape[0] / 2)

            unstable_neurons = list()

            for i in range(layer_numbers):
                lower_label = f"lower_{i}"
                upper_label = f"upper_{i}"

                bool_mask = (df[lower_label] < 0) & (df[upper_label] > 0)
                unstable_neurons.append(bool_mask.sum())
            temp_dict = {f"layer_{i}": unstable_neurons[i] for i in range(layer_numbers)}
            temp_dict["file_name"] = file
            results.append(temp_dict)

        to_write_df = pd.DataFrame(results)
        to_write_df.to_csv(output_file_path, index=False)

