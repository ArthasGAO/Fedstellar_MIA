import logging
import os
import sys
from DFL_util import dataset_model_set
from SIA import SIA
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation, CIFAR10DatasetExtendedAugmentation
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset, DataLoader

    
def sample_iid_train_data(dataset, num_client, data_size, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))

    np.random.shuffle(all_indices)

    total_data_points = num_client * data_size
    if total_data_points > len(dataset):
        raise ValueError("Not enough data points to create the required number of subsets.")

    subsets_indices = [all_indices[i * data_size:(i + 1) * data_size] for i in range(num_client)]
    subsets = [Subset(dataset, indices) for indices in subsets_indices]

    return subsets

def build_classes_dict(dataset):
    classes_dict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        classes_dict[label].append(idx)
    return classes_dict

def sample_dirichlet_train_data(dataset, num_client, data_size, alpha, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)

    subset_indices = all_indices[:data_size]
    subset_dataset = Subset(dataset, subset_indices)

    data_classes = build_classes_dict(subset_dataset)
    per_participant_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    for n in range(no_classes):
        current_class_size = len(data_classes[n])  # Use actual size of the current class
        np.random.shuffle(data_classes[n])
        sampled_probabilities = current_class_size * np.random.dirichlet(
            np.array(num_client * [alpha]))
        for user in range(num_client):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]

    for i in per_participant_list:
        actual_indices = [subset_indices[idx] for idx in
                          per_participant_list[i]]  # Map back to original dataset indices
        per_participant_list[i] = Subset(dataset, actual_indices)

    return per_participant_list
    
    
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    handler.setFormatter(formatter)

    return logger
    

def main():
    DATASET = ["Cifar10", "Cifar10extend"]
    MODEL = ["mlp", "cnn"]
    TOPOLOGY = ["fully"]

    IID = [0]
    ALPHA = 0.1
    
    ROUNDS = range(1, 11, 1) 
    
    CLIENT_IDX = range(0, 10, 1)
    
    NUM_CLIENTS = len(CLIENT_IDX)
    
    SEED = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to ensure which device we are using here.
    labels = [f"node{i+1}" for i in range(NUM_CLIENTS)] 
    
    backup_dataset = CIFAR10DatasetNoAugmentation()

    for dataset in DATASET:
        for model_name in MODEL:
            for topo in TOPOLOGY:
                for iid in IID:
                    file_name = dataset + "_" + model_name + "_" + topo + "_" + str(NUM_CLIENTS) + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(SEED)
                    sia_model_directory = f"./saved_models/{file_name}/Local_models"
                    
                    global_dataset, global_model = dataset_model_set(dataset, model_name)
                    
                    if global_dataset == None or global_model == None:
                        continue
                        
                    # global_model_class = type(global_model)
                    
                    if iid:
                        train_subsets = sample_iid_train_data(global_dataset.train_set, NUM_CLIENTS, 2500, SEED)
                        train_loaders = [DataLoader(i, batch_size=128, shuffle=False, num_workers=12) for i in train_subsets]
                    else:
                        # train_subsets = sample_dirichlet_train_data(global_dataset.train_set, NUM_CLIENTS, 2500*NUM_CLIENTS, ALPHA, SEED)
                        train_subsets = sample_dirichlet_train_data(backup_dataset.train_set, NUM_CLIENTS, 2500*NUM_CLIENTS, ALPHA, SEED)
                        train_loaders = [DataLoader(train_subsets[i], batch_size=128, shuffle=False, num_workers=12) for i in range(NUM_CLIENTS)]
                        
                    '''excel_path = f'./saved_result_sia/{file_name}.xlsx'
                    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:'''
                    
                    base_directory = f'./saved_result_sia/{file_name}_v1'
                    os.makedirs(base_directory, exist_ok=True)
                    
                    for rou in ROUNDS:
                        w_locals = []
                        
                        for client_idx in CLIENT_IDX:
                            # checkpoint_path = sia_model_directory + f"/Round_{rou}/client_{client_idx+1}.ckpt"
                            
                            model_path = sia_model_directory + f"/Round_{rou}/client_{client_idx+1}.pth"
                            state_dict = torch.load(model_path)
                            # model_local = global_model_class.load_from_checkpoint(checkpoint_path)  # Load the correct model
                            # w_local = model_local.state_dict()
                            # w_locals.append(w_local)
                            w_locals.append(state_dict)
                            
                        sia = SIA(w_locals, train_loaders) 
                        sia_res = sia.attack(global_model.to(device)) # a numpy array indicating the data samples source distribution
                        
                        df = pd.DataFrame(sia_res, index=labels, columns=labels)
                        
                        excel_file_path = os.path.join(base_directory, f'Round_{rou}.xlsx')
                        df.to_excel(excel_file_path, index=False)
                        
                        
                        
        
                                    
                        
                        

if __name__ == '__main__':
    main()