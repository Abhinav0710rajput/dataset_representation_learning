import torch
import torch.nn.functional as F
import random
from torch import nn, optim
import os
import pandas as pd
import argparse


from data_loader import *
from sampling import *
from model import *

def save_model(model, filepath, optimizer=None, epoch=None, loss=None, scheduler=None):

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'out_dim': model.out_dim
        }
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
    if loss is not None:
        save_dict['loss'] = loss
    if scheduler is not None:
        save_dict['scheduler'] = scheduler
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device='cpu'):

    checkpoint = torch.load(filepath, map_location=device)

    # Recreate model
    config = checkpoint['model_config']
    model = Dataset2Vec(
        input_dim=config['input_dim'],
        out_dim=config['out_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Extract optional saved data
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)
    scheduler = checkpoint.get('scheduler', None)

    #print(f"Model loaded from {filepath}")
    return model, optimizer_state, epoch, loss


######### PRE TRAINING #################################################


def compute_similarity(embedding1, embedding2, gamma=1.0):

    return torch.exp(- gamma * torch.norm(embedding1 - embedding2, dim=1))

def train_step_pt(model, dataset_list, batch_size=4, gamma=1.0, device='cpu', optimizer=None, scheduler=None):
    model.train()

    pos_pairs = []
    neg_pairs = []

    indices = random.sample(range(len(dataset_list)), batch_size)
    base_datasets = [dataset_list[i] for i in indices]

    simpos = 0 
    simneg = 0

    # Positive pairs
    for dataset in base_datasets:
        x1, y1 = sample_batch(*dataset)
        x2, y2 = sample_batch(*dataset)
        pos_pairs.append(((x1.to(device), y1.to(device)), (x2.to(device), y2.to(device))))

    # Negative pairs
    remaining_indices = list(set(range(len(dataset_list))) - set(indices))
    if len(remaining_indices) < batch_size:
        raise ValueError("Not enough disjoint datasets to form negative pairs")

    neg_indices = random.sample(remaining_indices, batch_size)
    neg_datasets = [dataset_list[i] for i in neg_indices]

    #print(indices, neg_indices) ################

    for i in range(batch_size):
        x1, y1 = sample_batch(*base_datasets[i])
        x2, y2 = sample_batch(*neg_datasets[i])
        neg_pairs.append(((x1.to(device), y1.to(device)), (x2.to(device), y2.to(device))))



    # Compute auxiliary loss
    losses = []
    for (x1, y1), (x2, y2) in pos_pairs:
        e1 = model(x1, y1).unsqueeze(0)
        e2 = model(x2, y2).unsqueeze(0)
        sim = compute_similarity(e1, e2, gamma)
        simpos = simpos+sim
        losses.append(-torch.log(sim))
    
    epsilon = 1e-7

    for (x1, y1), (x2, y2) in neg_pairs:
        e1 = model(x1, y1).unsqueeze(0) 
        e2 = model(x2, y2).unsqueeze(0) 
        sim = compute_similarity(e1, e2, gamma)
        """

        in case the sim is of very high (becuse e1 and e2 are of very small order) ~ 1, subtract a small epsilion 

        """
        simneg = simneg + sim
        sim = sim - epsilon 
        losses.append(-torch.log(1 - sim))

    print((simpos/len(pos_pairs)).item(), "positive pair similarity")
    print((simneg/len(neg_pairs)).item(), "negative pair similarity")

    loss = torch.mean(torch.cat(losses))

    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if scheduler:
        scheduler.step()

    return loss.item()


#########################################################


################## FINETUNING ###########################


def train_step(model, dataset_list, split_datasets, data, batch_size=4, device='cpu', optimizer=None, scheduler=None):
    model.train()

    fr_weightloss = torch.zeros(1, device=device)

    indices = random.sample(range(len(dataset_list)), batch_size)

    #Cache embeddings
    cache = {}
    for idx in indices:
        d = dataset_list[idx]
        x, y = d  # or sample_batch(*d)
        x = x.to(device)
        y = y.to(device)
        e = model(x, y) 
        cache[idx] = {
            "embedding": e.unsqueeze(0),
            "name": split_datasets[idx],  
        }

    #Precompute RS vectors 
    rs_cache = {}
    for idx in indices:
        dname = cache[idx]["name"]
        rs = torch.tensor(
            data[data['Dataset'] == dname][['DT', 'RF', 'SVM', 'NB', 'KNN']].values,
            dtype=torch.float32,
            device=device
        )
        rs_cache[idx] = rs

    #FR loss
    fr_loss = torch.zeros(1, device=device)
    for i, idx_i in enumerate(indices):
        e1 = cache[idx_i]["embedding"]
        rs_i = rs_cache[idx_i]

        for idx_j in indices[i+1:]:
            e2 = cache[idx_j]["embedding"]
            rs_j = rs_cache[idx_j]

            d_embed = torch.norm(e1 - e2, p=2)
            d_rs = torch.norm(rs_i - rs_j, p=2)

            fr_loss += (d_embed - d_rs) ** 2
            
    loss = fr_loss/(batch_size*(batch_size-1)/2) 

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

    
    device_id = 5
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    print(f"\n Using device: {device} — {torch.cuda.get_device_name(device_id)}")

    device = 'cpu'

    mode = args.mode

    epoch_pt = 10000
    epoch_ft = 10000

    ################################## 

    type = mode

    output_dim = 32
    
    pt = True  # pretraining
    ft = True  # finetuning
    resume = True 

    if mode == 'pt':
        ft = False
    if mode == 'ft':
        pt = False
    if mode == 'ft-scratch':
        resume = False
    if mode == 'dt':
        pt = False
        resume = False

    if type in ['ft', 'ft-scratch']:
        type = 'ft'

    splits = args.split

    for split in splits:    ####### for split in splits
        
        split_datasets = load_dataset(split)
        dataset_list = load_all_datasets(data_names=split_datasets)
        print(f"Loaded {len(dataset_list)} datasets | Split {split}")


        model = Dataset2Vec(input_dim=2, out_dim=output_dim).to(device)

        if pt:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss_pretraining = []
            for epoch in range(1, epoch_pt+1):
                loss = train_step_pt(model, dataset_list, batch_size=32, gamma=1.0, device=device, optimizer=optimizer)
                loss_pretraining.append(loss)


                print(f"Epoch {epoch:03d} | Auxiliary Loss: {loss:.4f}")

    
                file_name = f"checkpoint/pretrained/losses/pt_out_{output_dim}_split_{split}.txt"
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(f"checkpoint/pretrained/losses/pt_out_{output_dim}_split_{split}.txt", "a") as f:
                    f.write(f"Epoch {epoch:03d} loss: {loss:.4f}\n")

                #### save the model, optimizer, epoch -> name of model pt_epoch_epoch_f_f_dim_g_g_dim_output_output_dim
                if(epoch%100 == 0):
                    print(f"Epoch {epoch:03d} | Auxiliary Loss: {loss:.4f}")
                    model_filename = f"checkpoint/pretrained/pt_epoch_{epoch}_out_{output_dim}_split_{split}.pth"
                    save_model(model, model_filename, optimizer, epoch, loss)

        
        ###### frobenious training ######
        if ft:

            if resume:
                best_model = f"checkpoint/pretrained/pt_epoch_10000_out_32_split_{split}.pth"
                model, _, _, _ = load_model(best_model, device)


            data = pd.read_csv("ft_dataset/classifiers_performance.csv")
            print("fine tuning dataset loaded")
            optimizer = optim.Adam(model.parameters(), lr=1e-3)


            for epoch in range(1, epoch_ft+1):

                loss = train_step(model, dataset_list, split_datasets, data, batch_size=32, device=device, optimizer=optimizer)

                print(f"Epoch {epoch:03d} | Frobenious_loss: {loss:.4f}")
                #save training info (print statement) in a txt file

                # SAVE LOSS LOGGING
                file_name = f"checkpoint/{type}/losses/{type}_out_{output_dim}_split_{split}.txt"
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(f"checkpoint/{type}/losses/{type}_out_{output_dim}_split_{split}.txt", "a") as f:
                    f.write(f"Epoch {epoch:03d} Frobenious_loss: {loss:.4f}\n")

                if(epoch%100 == 0):    
                    #### save the model 
                    print(f"Epoch {epoch:03d} | Frobenious_loss: {loss:.4f}")
                    print("saving")
                    model_filename = f"checkpoint/{type}/{type}_epoch_{epoch}_out_{output_dim}_split_{split}.pth"
                    save_model(model, model_filename, optimizer, epoch, loss)

        

        
"""

. ~/.bashrc; conda activate torchenv; python train_fr.py --split 1 2 3 4 5 --mode ft

"""









