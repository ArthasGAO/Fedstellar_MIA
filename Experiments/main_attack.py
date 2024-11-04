import logging
import os
import sys
from DFL_util import dataset_model_set, prepare_evaluation_dataset1, prepare_shadow_dataset, prepare_evaluation_dataset2, prepare_dirichlet_eval_data, prepare_iid_eval_data
from ShadowModelMIA_FL import ShadowModelBasedAttack
from MIA_FL import MembershipInferenceAttack
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation, CIFAR10DatasetExtendedAugmentation
from MetricMIA_FL import MetricBasedAttack
from ClassMetricMIA_FL import ClassMetricBasedAttack
import torch
import time

    
    
def main():
    DATASET = ["Cifar10no"]
    MODEL = ["cnn", "mlp"]
    TOPOLOGY = ["fully"]#, "star", "ring"]#, "random"]

    IID = [1]
    ALPHA = 0.1
    
    ROUNDS = range(1, 11, 1) 
    EPOCHS_DICT = {1:9, 2:19, 3:29, 4:39, 5:49, 6:59, 7:69, 8:79, 9:89, 10:99}
    
    CLIENT_IDX = range(1, 11, 1)
    
    NUM_CLIENTS = len(CLIENT_IDX)
    
    SEED = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to ensure which device we are using here.
    
    logger_info = {"table_dir":None, "Adversary":None, "Model":None, "iid":None, "topo":None, "Node":None, "Round":None }
    
    
    # backup_dataset = CIFAR10DatasetNoAugmentation()

    for dataset in DATASET:
        for model_name in MODEL:
            
            global_dataset, global_model = dataset_model_set(dataset, model_name)
            
            if global_dataset == None or global_model == None:
                continue
                
            shadow_train_loader, shadow_test_loader = prepare_shadow_dataset(global_dataset, 2500*NUM_CLIENTS, SEED)
                
            for topo in TOPOLOGY:
                logger_info["topo"] = topo
                
                for iid in IID:
                    logger_info["iid"] = iid
                    
                    file_name = dataset + "_" + model_name + "_" + topo + "_" + str(NUM_CLIENTS) + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(SEED)
                    shadow_file_name = dataset + "_" + model_name + "_" + "25000" + "_" + str(SEED)
                    
                    if iid:
                        in_eval_dataloader, out_eval_dataloader, indexing_map = prepare_iid_eval_data(global_dataset, NUM_CLIENTS, 2500, SEED)
                        # in_eval_dataloader, out_eval_dataloader, indexing_map = prepare_iid_eval_data(backup_dataset, NUM_CLIENTS, 2500, SEED)
                        
                    else:
                        in_eval_dataloader, out_eval_dataloader, indexing_map = prepare_dirichlet_eval_data(global_dataset, NUM_CLIENTS, 2500*NUM_CLIENTS, ALPHA, SEED)
                    
                    
                    for rou in ROUNDS:
                        logger_info["Round"] = f"Round_{rou}"
                        
                        for idx in CLIENT_IDX:
                            logger_info["Node"] = f"Node_{idx}"
                            model_path = f"./saved_models/{file_name}/Aggregated_models/Round_{rou}/client_{idx}.pth"
                            
                            
                            state_dict = torch.load(model_path)
                            global_model.load_state_dict(state_dict)  # Load the correct model
                            
                            logger_info["Adversary"] = "Adversary 1"
                            logger_info["table_dir"] = f"./saved_results/{dataset}/shadow_attack_fl_0.xlsx"
                            
                            print(EPOCHS_DICT[rou])
                            attack1 = ShadowModelBasedAttack(global_model.to(device), global_dataset,in_eval_dataloader, out_eval_dataloader, logger_info, indexing_map, EPOCHS_DICT[rou]                                                          ,[shadow_train_loader], [shadow_test_loader], 1, "Test", shadow_file_name)
                            attack1._generate_attack_dataset()
                            attack1.MIA_shadow_model_attack()
                            
                            logger_info["table_dir"] = f"./saved_results/{dataset}/class_metric_fl_0.xlsx"
                            c_attack1 = ClassMetricBasedAttack(global_model.to(device), global_dataset,in_eval_dataloader, out_eval_dataloader, logger_info, indexing_map,
                                                                attack1.shadow_train_res, attack1.shadow_test_res)
                            c_attack1._mem_inf_benchmarks()
                            
                            print(f"Node_{idx} done!")
                            
                            if topo == "fully":
                                break
                        print(f"Round_{rou} done!")
                        
                        

if __name__ == '__main__':
    main()
