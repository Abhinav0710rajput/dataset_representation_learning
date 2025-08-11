import torch
import torch.nn.functional as F
import random
from torch import nn, optim
import os
import pandas as pd
import argparse
from data_loader import *
from sampling import *
from model_rs import *
from train_fr import save_model, load_model


################## FINETUNING/DT ###########################

def train_step(model, dataset_list, split_datasets, data, batch_size=4, device='cpu', optimizer=None, scheduler=None):
    model.train()

    indices = random.sample(range(len(dataset_list)), batch_size)
    loss_fn = torch.nn.MSELoss()

    loss = torch.zeros(1, device=device)

    for i in indices:
        x1, y1 = dataset_list[i]
        x1 = x1.to(device)
        y1 = y1.to(device)
        embedding, result = model(x1, y1)
        result = result.unsqueeze(0)

        d1name = split_datasets[i]
        rs_i = torch.tensor(data[data['Dataset'] == d1name][['DT', 'RF', 'SVM', 'NB', 'KNN']].values,dtype=torch.float32,device=device)

        print(result.shape, rs_i.shape)

        loss += loss_fn(result, rs_i)
            
    loss = loss/batch_size


    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if scheduler:
        scheduler.step()

    return loss.item()

########################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Dataset2Vec model.")
    parser.add_argument(
        '--split',
        type=int,
        nargs='+',  # allows multiple values
        default=None,
        help='Specify one or more splits to run (e.g., --split 1 2 5)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['dt', 'ft'],  # restricts to specific valid modes
        default=None,
        help='Specify the mode: "dt" for direct training, "ft" for fine-tuning (default: ft)'
    )

    args = parser.parse_args()
    mode = args.mode

    epochs = 10000
    output_dim = 32

    #########################################
 

    device_id = 5
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    print(f"\n Using device: {device} — {torch.cuda.get_device_name(device_id)}")
    #device = 'cpu'


    ##################################

    splits = args.split

    for split in splits:   
        
        split_datasets = load_dataset(split)
        dataset_list = load_all_datasets(data_names=split_datasets)
        print(f"Loaded {len(dataset_list)} datasets | Split {split}")
        
        ###### result space training ######

        if mode == 'ft':
            embedder_loc = f"checkpoint/pretrained/pt_epoch_10000_out_32_split_{split}.pth"
            embedder, _, _, _ = load_model(embedder_loc, device)
            embedder = embedder.to(device)
            model = d2v_rs(embedder)
        else:
            model = d2v_rs(None)
        
        # Move entire model to device
        model = model.to(device)


        data = pd.read_csv("ft_dataset/classifiers_performance.csv")
        print("fine tuning dataset loaded")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)


        for epoch in range(1, epochs+1):

            loss = train_step(model, dataset_list, split_datasets, data, batch_size=32, device=device, optimizer=optimizer)
            
            print(f"Epoch {epoch:03d} | Result Space Loss: {loss:.4f}")
            #save training info (print statement) in a txt file

            # SAVE LOSS LOGGING
            file_name = f"checkpoint/rs_{mode}/losses/rs_{mode}_out_{output_dim}_split_{split}.txt"
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "a") as f:
                f.write(f"Epoch {epoch:03d} Result Space Loss: {loss:.4f}\n")

            if(epoch%100 == 0):    
                #### save the model 
                print(f"Epoch {epoch:03d} | Result Space Loss: {loss:.4f}")
                print("saving")
                model_filename = f"checkpoint/rs_{mode}/rs_{mode}_epoch_{epoch}_out_{output_dim}_split_{split}.pth"
                checkpoint = model.embedder
                save_model(checkpoint, model_filename, optimizer, epoch, loss)

        

        
"""

. ~/.bashrc; conda activate torchenv; python train_rs.py --split 1 2 5 --mode ft

"""








