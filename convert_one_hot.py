import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder

base_path = 'datasets/'

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    label_file = os.path.join(folder_path, 'labels_py.dat')

    if os.path.isfile(label_file):
        # Read original labels
        with open(label_file, 'r') as f:
            labels = [int(line.strip()) for line in f.readlines()]

        labels_np = np.array(labels).reshape(-1, 1)

        # One-hot encode
        encoder = OneHotEncoder(sparse=False)
        onehot_labels = encoder.fit_transform(labels_np)

        # Overwrite the original file with one-hot encoded lines
        with open(label_file, 'w') as f:
            for row in onehot_labels:
                line = ' '.join(map(str, row.astype(int)))
                f.write(f"{line}\n")

        print(f"[✓] Overwritten {label_file} with one-hot encoded labels.")
import os

base_path = 'datasets/'

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    label_file = os.path.join(folder_path, 'labels_py.dat')

    if os.path.isfile(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Replace spaces with commas
        new_lines = [line.strip().replace(' ', ',') + '\n' for line in lines]

        with open(label_file, 'w') as f:
            f.writelines(new_lines)

        print(f"[✓] Converted space-separated to comma-separated in {label_file}")
