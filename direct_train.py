import torch
import torch.nn.functional as F
import random
from torch import nn, optim
import os
import pandas as pd


from data_loader import *
from sampling import *
from model import *

from train import *



pretrain = False
epoch_pt = 10000
epoch_ft = 10000
f_dim = 64
g_dim = 64
output_dim = 32

splits = [5]


for split in splits:
    
    split_datasets = load_dataset(split)
    dataset_list = load_all_datasets(data_names=split_datasets)
    print(f"Loaded {len(dataset_list)} datasets | Split {split}")

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(f"cuda:{4}")
    print(f"Using device: {device}")


    model = Dataset2Vec(input_dim=2, f_dim=f_dim, g_dim=g_dim, out_dim=output_dim).to(device)
    
    ###### frobenious training ######
    data = pd.read_csv("ft_dataset/classifiers_performance.csv")
    print("fine tuning dataset loaded")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_ft = []
    for epoch in range(1, epoch_ft):

        loss = train_step(model, dataset_list, split_datasets, data, batch_size=32, gamma=0.01, device=device, optimizer=optimizer)
        loss_ft.append(loss)
        print(f"Epoch {epoch:03d} | Frobenious_loss: {loss:.4f}")

        file_name = f"checkpoint/direct_trained/losses/direct_fr_{f_dim}_g_{g_dim}_out_{output_dim}_split_{split}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "a") as f:
            f.write(f"Epoch {epoch:03d} Frobenious_loss: {loss:.4f}\n")

        if(epoch%(epoch_ft/100) == 0):    
            #### save the model 
            print("saving")
            model_filename = f"checkpoint/direct_trained/direct_fr_epoch_{epoch}_f_{f_dim}_g_{g_dim}_out_{output_dim}_split_{split}.pth"
            save_model(model, model_filename, optimizer, epoch, loss)