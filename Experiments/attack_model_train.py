import json
import logging
import os
from lightning import Trainer
from torch.utils.data import DataLoader
from MyCustomCallbackShadow import MyCustomCheckpointShadow
from DFL_util import parse_experiment_file, dataset_model_set
import sys
import numpy as np
from torch.utils.data import Subset, DataLoader, ConcatDataset
import torch
from ShadowModelMIA_FL import ShadowModelBasedAttack



def prepare_shadow_dataset(global_dataset, size, seed):
    """This function uses the unused train dataset as the shadow test."""
    np.random.seed(seed)
    all_train_indices = np.arange(len(global_dataset.train_set))

    np.random.shuffle(all_train_indices)
    
    if size > 12500:
        shadow_train_indices = all_train_indices[size:2*size] # 25000:50000
        shadow_train_dataset = Subset(global_dataset.train_set, shadow_train_indices)
        
        first_shadow_test_dataset = global_dataset.test_set
        extra_shadow_test_indices = np.random.choice(all_train_indices[:size], size=size - len(global_dataset.test_set),
                                                     replace=False)
                                                     
        extra_shadow_test_dataset = Subset(global_dataset.train_set, extra_shadow_test_indices)
        shadow_test_dataset = ConcatDataset([first_shadow_test_dataset, extra_shadow_test_dataset])
    else:
        shadow_train_indices = all_train_indices[2*size:3*size] 
        shadow_train_dataset = Subset(global_dataset.train_set, shadow_train_indices)
        
        shadow_test_indices = all_train_indices[3*size:4*size]
        shadow_test_dataset = Subset(global_dataset.train_set, shadow_test_indices)

    shadow_train_dataloader = DataLoader(shadow_train_dataset, batch_size=128, shuffle=True, num_workers=12)
    shadow_test_dataloader = DataLoader(shadow_test_dataset, batch_size=128, shuffle=False, num_workers=12)

    return shadow_train_dataloader, shadow_test_dataloader

'''Since for the federation case, the shadow model's training dataset should be disjoint from the federation training set.
   Thus, actually, for each client, the shadow model can be used jointly without violating the shadow model's convention.
   To faciliate evaluating, we train one shadow model satisfying requirements and let it be used seperately from attack.'''
   
   
def main():
    DATASET = ["Cifar10no", "Cifar10", "Cifar10extend", "Mnist"]
    MODEL = ["cnn", "mlp"]
    SIZE = [25000]
    MAX_EPOCHS = 99
    EPOCHS = range(9, 100, 10)
    SEED = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset in DATASET:
        for model in MODEL:
            for size in SIZE:    
                file_name = dataset + "_" + model + "_" + str(size) + "_"  + str(SEED)
                
                global_dataset, global_model = dataset_model_set(dataset, model)
        
                if global_dataset == None or global_model == None:
                    continue
                
                shadow_train_loader, shadow_test_loader = prepare_shadow_dataset(global_dataset, size, SEED)
                
                for epoch in EPOCHS:
        
                    shadow_attack1 = ShadowModelBasedAttack(global_model.to(device), global_dataset, shadow_train_loader, shadow_test_loader, 1,1,
                                                                epoch, [shadow_train_loader], [shadow_test_loader], 1, "Train", file_name)
                    shadow_attack1._generate_attack_dataset()
                    shadow_attack1.MIA_shadow_model_attack()
            


if __name__ == '__main__':
    main()
