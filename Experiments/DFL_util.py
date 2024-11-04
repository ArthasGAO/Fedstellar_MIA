import copy
import json
import os
from collections import Counter
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation, CIFAR10DatasetExtendedAugmentation
from Dataset.mnist import MNISTDataset
from Dataset.fmnist import FashionMNISTDataset
from Model.mlp import MLP
from Model.cnn import CIFAR10ModelCNN
from Model.cnn_fmnist import FMNISTModelCNN
from torch.utils.data import Subset, DataLoader, ConcatDataset
from Node import Node
import numpy as np
import torch
from collections import defaultdict


def dataset_model_set(dataset, model):
    g_dataset = None
    g_model = None
    if dataset == "Mnist":
        g_dataset = MNISTDataset()
        if model == "mlp":
            g_model = MLP(784, 1e-3, 10)
        elif model == "cnn":
            pass
        else:
            pass
    elif dataset == "FMnist":
        g_dataset = FashionMNISTDataset()
        if model == "mlp":
            g_model = MLP(784, 1e-3, 10)
        elif model == "cnn":
            pass
        else:
            pass
    elif dataset == "Cifar10":
        g_dataset = CIFAR10Dataset()
        if model == "mlp":
            pass
        elif model == "cnn":
            g_model = CIFAR10ModelCNN()
    elif dataset == "Cifar10no":
        g_dataset = CIFAR10DatasetNoAugmentation()
        if model == "mlp":
            pass
        elif model == "cnn":
            g_model = CIFAR10ModelCNN()
    elif dataset == 'Cifar10extend':
        g_dataset = CIFAR10DatasetExtendedAugmentation()
        if model == "mlp":
            pass
        elif model == "cnn":
            g_model = CIFAR10ModelCNN()
    return g_dataset, g_model
    
    
def parse_experiment_file(file_path):
    experiment_info = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into key and value parts
            parts = line.strip().split(':')
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                # Check if the value is intended to be a list
                if value.startswith('[') and value.endswith(']'):
                    # Parse the list value using json.loads after replacing single quotes with double quotes
                    # This is necessary because json expects double-quoted strings
                    value = json.loads(value.replace("'", '"'))
                experiment_info[key] = value

    return experiment_info
    

def create_nodes_list(num, datamodule, model):
    node_list = []
    for i in range(0, num):
        node = Node(i + 1, copy.deepcopy(model), datamodule[0][i], datamodule[1][i])
        node_list.append(node)
    return node_list
    

def create_adjacency(nodes_list, topology):
    num_nodes = len(nodes_list)

    # random.shuffle(nodes_list)  # include randomness

    if topology == "fully":
        for node in nodes_list:
            node.neigh = {s for s in nodes_list if s != node}
    elif topology == "star":
        root = nodes_list[0]
        nodes_list[0].neigh = {s for s in nodes_list if s != root}
        for node in nodes_list:
            if node != root:
                node.neigh = {root}
    elif topology == "ring":
        for i, node in enumerate(nodes_list):
            left_neighbor = nodes_list[(i - 1) % num_nodes]
            right_neighbor = nodes_list[(i + 1) % num_nodes]
            node.neigh = {left_neighbor, right_neighbor}
    else:
        raise ValueError("Unsupported topology")

    return nodes_list
    

def create_custom_adjacency(nodes_list):
    # Nodes 0
    nodes_list[0].neigh.add(nodes_list[4])
    # Nodes 4
    nodes_list[4].neigh.add(nodes_list[0])
    nodes_list[4].neigh.add(nodes_list[5])
    # Nodes 5
    nodes_list[5].neigh.add(nodes_list[4])
    nodes_list[5].neigh.add(nodes_list[3])
    nodes_list[5].neigh.add(nodes_list[8])
    nodes_list[5].neigh.add(nodes_list[9])
    nodes_list[5].neigh.add(nodes_list[2])
    # Nodes 3
    nodes_list[3].neigh.add(nodes_list[5])
    # Nodes 8
    nodes_list[8].neigh.add(nodes_list[5])
    nodes_list[8].neigh.add(nodes_list[9])
    # Nodes 9
    nodes_list[9].neigh.add(nodes_list[8])
    nodes_list[9].neigh.add(nodes_list[5])
    # Nodes 2
    nodes_list[2].neigh.add(nodes_list[5])
    nodes_list[2].neigh.add(nodes_list[6])
    # Nodes 6
    nodes_list[6].neigh.add(nodes_list[2])
    nodes_list[6].neigh.add(nodes_list[1])
    nodes_list[6].neigh.add(nodes_list[7])
    # Nodes 7
    nodes_list[7].neigh.add(nodes_list[6])
    # Nodes 1
    nodes_list[1].neigh.add(nodes_list[6])

    return nodes_list
    

def sample_iid_train_data(dataset, num_client, data_size, seed, logger):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset))

    np.random.shuffle(all_indices)
    logger.info(f"All dataset indices: {all_indices}:")

    total_data_points = num_client * data_size
    if total_data_points > len(dataset):
        raise ValueError("Not enough data points to create the required number of subsets.")

    subsets_indices = [all_indices[i * data_size:(i + 1) * data_size] for i in range(num_client)]
    subsets = [Subset(dataset, indices) for indices in subsets_indices]

    net_dataidx_map = {i: subsets_indices[i] for i in range(num_client)}

    for i, subset_indices in net_dataidx_map.items():
        label_counter = Counter()
        for idx in subset_indices:
            _, label = dataset[idx]
            label_counter[label] += 1

        # Ensure every class is represented, even if it has zero samples
        all_classes_distribution = {label: label_counter.get(label, 0) for label in
                                    range(10)}  # Assuming dataset labels are from 0 to 9
        total_samples = sum(label_counter.values())
        logger.info(
            f"Participant {i}: Total Data Samples - {total_samples}, Label Distribution - {all_classes_distribution}")

    return subsets
    

def build_classes_dict(dataset):
    classes_dict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        classes_dict[label].append(idx)
    return classes_dict


def sample_dirichlet_train_data(dataset, num_client, data_size, alpha, seed, logger):
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

    for idx, subset in per_participant_list.items():
        label_counter = Counter({label: 0 for label in range(no_classes)})
        for _, label in subset:
            label_counter[label] += 1
        total_samples = sum(label_counter.values())
        logger.info(f"Participant {idx}: Total Data Samples - {total_samples}")
        logger.info(f"Label Distribution: {dict(label_counter)}")

    return per_participant_list


def save_params(params_dict, round_num, file_name, client_id=None, is_global=False):
    if is_global:
        directory = f"saved_models/{file_name}/Aggregated_models/Round_{round_num}"
        os.makedirs(directory, exist_ok=True)
        filename = f"client_{client_id}.pth"
    else:
        directory = f"saved_models/{file_name}/Nei_aggregated_models/Round_{round_num}"
        os.makedirs(directory, exist_ok=True)
        filename = f"client_{client_id}.pth"
        
    path = os.path.join(directory, filename)
    torch.save(params_dict, path)
    
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
    
    
def prepare_evaluation_dataset1(dataset, size, num_client, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))

    np.random.shuffle(all_indices)

    subsets_indices = [all_indices[i * size:(i + 1) * size] for i in range(num_client)]
    subsets = [Subset(dataset.train_set, indices) for indices in subsets_indices]
    
    in_eval_loader = [DataLoader(subsets[i], batch_size=128, shuffle=False, num_workers=12)
                         for i in range(num_client)]
                         
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[size*num_client:]),batch_size=128, shuffle=False, num_workers=12) # this is for the 25000 case
    
    return in_eval_loader, out_eval_loader


def prepare_evaluation_dataset2(dataset, size, num_client, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))
    np.random.shuffle(all_indices)

    # Create subsets indices for in-eval loaders
    subsets_indices = [all_indices[i * size:(i + 1) * size] for i in range(num_client)]
    combined_indices = np.concatenate(subsets_indices)
    
    # Create the combined dataset and dataloader for in-eval
    combined_subset = Subset(dataset.train_set, combined_indices)
    combined_in_eval_loader = DataLoader(combined_subset, batch_size=128, shuffle=False, num_workers=12)
    
    # Create the out-eval dataloader for the remaining data
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[size * num_client:]), batch_size=128, shuffle=False, num_workers=12)
    
    return combined_in_eval_loader, out_eval_loader
    

def prepare_dirichlet_eval_data(dataset, num_client, data_size, alpha, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))
    np.random.shuffle(all_indices)

    subset_indices = all_indices[:data_size]
    subset_dataset = Subset(dataset.train_set, subset_indices)

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
            
    node_data = {}
    for i in per_participant_list:
      node_data[i] = len(per_participant_list[i])

    start_end_indices = []

    # Initialize the start index
    start_idx = 0

    # Iterate through the dictionary to calculate the start and end indices
    for key, value in node_data.items():
        end_idx = start_idx + value
        start_end_indices.append((start_idx, end_idx))
        # Update the start index for the next iteration
        start_idx = end_idx
            
    for i in per_participant_list:
        actual_indices = [subset_indices[idx] for idx in
                          per_participant_list[i]]  # Map back to original dataset indices
        per_participant_list[i] = actual_indices
    
    combined_indices = []   
    for i in range(num_client):
        combined_indices.append(per_participant_list[i])
    combined_indices = np.concatenate(combined_indices)
      
    combined_subset = Subset(dataset.train_set, combined_indices)
    combined_in_eval_loader = DataLoader(combined_subset, batch_size=128, shuffle=False, num_workers=12)  
    
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[data_size:2*data_size]), batch_size=128, shuffle=False, num_workers=12) # use non-training samples as out

    return combined_in_eval_loader, out_eval_loader, start_end_indices
    
def prepare_iid_eval_data(dataset, num_client, data_size, seed):
    size = data_size * num_client
    
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))

    np.random.shuffle(all_indices)
    
    combined_indices = all_indices[:size]
    combined_subset = Subset(dataset.train_set, combined_indices)
    combined_in_eval_loader = DataLoader(combined_subset, batch_size=128, shuffle=False, num_workers=12)
    
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[size:2 * size]), batch_size=128, shuffle=False, num_workers=12) # use non-training samples as out

    return  combined_in_eval_loader, out_eval_loader, [(i * data_size, (i + 1) * data_size) for i in range(num_client)]






    