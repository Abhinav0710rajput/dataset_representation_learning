from tkinter import E
import torch
import os
import re
from data_loader import *
from train import *
import csv
import shutil

# Clear the contents of 'embeddings' and 'full_embeddings' folders
for folder in ['embeddings', 'full_embeddings']:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


import os
import re
import csv

base = "checkpoint/"
pattern = re.compile(r'(\d+)\.pth$')  # Matches digits before `.pth` at the end

for type in ['ft', 'dt', 'pretrained']:
    path = os.path.join(base, type)

    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        continue

    print(f"\n--- {type.upper()} ---")

    for file_name in sorted(os.listdir(path), reverse=True):  # <-- Reverse alphabetical order
        if file_name.endswith(".pth"):
            match = pattern.search(file_name)
            if match:
                split = int(match.group(1))  # Extract the split number
                print(f"File: {file_name} | Split: {split}")

                dataset_names = load_dataset(split, embedding_gen=True)
                datasets = load_all_datasets(dataset_names)
                checkpoint_path = os.path.join(path, file_name)

                device = 'cpu'
                model, optimizer_state, epoch, loss = load_model(checkpoint_path, device=device)

                embeddings = [] 
                model.eval()

                for data in datasets:
                    X, Y = data
                    e = model(X, Y).unsqueeze(0)
                    embeddings.append(e)

                embedding_path = 'embeddings/'
                os.makedirs(embedding_path, exist_ok=True)

                # Save embeddings as CSV
                csv_file = os.path.join(embedding_path, file_name.replace(".pth", ".csv"))

                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ["Dataset"] + [str(i) for i in range(embeddings[0].shape[1])]
                    writer.writerow(header)

                    for name, embedding in zip(dataset_names, embeddings):
                        writer.writerow([name] + embedding.squeeze(0).tolist())

                print(f"Saved embeddings to: {csv_file}")
                print(checkpoint_path)

            else:
                print(f"Could not extract split number from: {file_name}")




import os
import pandas as pd
import re
from collections import defaultdict

# Set paths
input_folder = "embeddings"              # Where the split files are
output_folder = "full_embeddings"        # Where you want to save merged files

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Group split files
split_pattern = re.compile(r'^(.*)_split_(\d+)\.csv$')
grouped_files = defaultdict(dict)

# Step 1: Organize split files
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        match = split_pattern.match(file)
        if match:
            config_base = match.group(1)
            split_num = int(match.group(2))
            grouped_files[config_base][split_num] = file

# Step 2: Merge only if all 5 splits (1–5) are present
for config_base, splits_dict in grouped_files.items():
    if all(i in splits_dict for i in range(1, 6)):
        dfs = []
        for i in range(1, 6):
            file_path = os.path.join(input_folder, splits_dict[i])
            df = pd.read_csv(file_path)
            dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True)

        output_filename = f"{config_base}.csv"
        output_path = os.path.join(output_folder, output_filename)
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Merged and saved: {output_filename}")
    else:
        print(f"❌ Skipping {config_base} not all 5 splits found.")





