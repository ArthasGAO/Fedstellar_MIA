import sys
from DFL_util import dataset_model_set, prepare_evaluation_dataset1, prepare_shadow_dataset
import torch

def check_state_dicts_identical(state_dict1, state_dict2):
    for key in state_dict1:
        if key not in state_dict2:
            print(f"Parameter {key} is not present in both models.")
            return False
        if not torch.allclose(state_dict1[key], state_dict2[key]):
            print(f"Parameter {key} is not identical.")
            return False
    return True
    
def ensure_all_state_dicts_identical(state_dicts):
    if not state_dicts:
        print("The list of state_dicts is empty.")
        return False

    reference_state_dict = state_dicts[0]
    for i, state_dict in enumerate(state_dicts[1:], start=1):
        if not check_state_dicts_identical(reference_state_dict, state_dict):
            print(f"State dictionary at index 0 and index {i} are not identical.")
            return False

    print("All state dictionaries are identical.")
    return True
    

def main():
    DATASET = ["Cifar10no", "Cifar10", "Cifar10extend", "Mnist"]
    MODEL = ["cnn", "mlp"]
    TOPOLOGY = ["fully"]

    IID = [1, 0]
    ALPHA = 0.1
    
    ROUNDS = range(1, 11, 1) 
    
    CLIENT_IDX = range(1, 11, 1)
    
    NUM_CLIENTS = len(CLIENT_IDX)
    
    SEED = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to ensure which device we are using here.
    
    # backup_dataset = CIFAR10DatasetNoAugmentation()

    for dataset in DATASET:
        for model_name in MODEL:
            global_dataset, global_model = dataset_model_set(dataset, model_name)
            
            if global_dataset == None or global_model == None:
                continue
        
            for topo in TOPOLOGY:
                for iid in IID:
                    file_name = dataset + "_" + model_name + "_" + topo + "_" + str(NUM_CLIENTS) + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(SEED)
                    
                    for rou in ROUNDS:
                        
                        rou_model_res = []
                        
                        for idx in CLIENT_IDX:
                        
                            model_path = f"./saved_models/{file_name}/Aggregated_models/Round_{rou}/client_{idx}.pth"
                            
                            
                            state_dict = torch.load(model_path)
                            
                            rou_model_res.append(state_dict)
                            
                        result = ensure_all_state_dicts_identical(rou_model_res)
                        print("Result:", result)    
                                
                            
                        
                        
                        


                        
                        
                        

if __name__ == '__main__':
    main()