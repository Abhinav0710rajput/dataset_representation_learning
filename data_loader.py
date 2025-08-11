import os
import torch
import numpy as np


def load_all_datasets(data_names, base_path='datasets/'):
    """
    Loads all datasets from subfolders inside base_path, but only those whose names are in data_names.
    Each subfolder must contain: *_py.dat and labels_py.dat
    Returns: list of (X, Y) as torch tensors
    """
    dataset_list = []

    for name in data_names:
        folder_path = os.path.join(base_path, name)

        try:
            x_file = [f for f in os.listdir(folder_path) if f.endswith('_py.dat') and 'labels' not in f][0]
            y_file = 'labels_py.dat'

            x_path = os.path.join(folder_path, x_file)
            y_path = os.path.join(folder_path, y_file)

            # Load with comma delimiter
            X = torch.tensor(np.loadtxt(x_path, dtype=np.float32, delimiter=','))
            Y = torch.tensor(np.loadtxt(y_path, dtype=np.float32, delimiter=','))

            # Ensure 2D shape
            if len(X.shape) == 1:
                X = X.unsqueeze(1)
            if len(Y.shape) == 1:
                Y = Y.unsqueeze(1)

            dataset_list.append((X, Y))
        except Exception as e:
            print(f"Skipping {folder_path} due to error: {e}")
            continue

    return dataset_list


def load_dataset(split=1, embedding_gen=False):
    dataset_names = []

    if not embedding_gen:
        for n in range (5):
            if(n+1 != split):
                txt_file = f"splits/s{n+1}.txt"
                try:
                    with open(txt_file, 'r') as f:
                        for line in f:
                            name = line.strip()
                            if name:
                                dataset_names.append(name)
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")
    else:
        txt_file = f"splits/s{split}.txt"
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        dataset_names.append(name)
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")

    return dataset_names

if __name__ == "__main__":
    
    for split in [1,2,3,4,5]:
        print(f"Loading datasets for split {split}")
        a = load_dataset(split)

        print("-----------------")
        print(a)
        print(len(a))
        m = load_all_datasets(a)
        print(a[23])
        print(m[23])
        print(len(m))

            



