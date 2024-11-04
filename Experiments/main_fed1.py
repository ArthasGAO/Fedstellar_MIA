import logging
import os
import sys
from torch.utils.data import DataLoader
from DFL_util import parse_experiment_file, dataset_model_set, create_nodes_list, \
    create_adjacency, sample_iid_train_data, sample_dirichlet_train_data, create_custom_adjacency, \
    save_params
from lightning import Trainer
from MyCustomCallback import MyCustomCheckpoint
import json


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
    TOPOLOGY = ["random"]#"fully", "star", "ring", "random"]
    ROUND = 10
    NUM_CLIENTS = 10
    DATASET = ["Cifar10no"]#, "Cifar10", "Cifar10extend", "Mnist"]
    MODEL = ["cnn", "mlp"]
    IID = [1]
    BATCH_SIZE = 128
    SIZE = 2500  # fixed as 2500
    MAX_EPOCHS = 10  # fixed as 10
    SEED = 42
    ALPHA = 0.1
    
    
    
    for dataset in DATASET:
        for model_name in MODEL:
            # dataset and model setting
            global_dataset, global_model = dataset_model_set(dataset, model_name)
            
            if global_dataset == None or global_model == None:
                continue
                
            for topo in TOPOLOGY:
                for iid in IID:
                    file_name = dataset + "_" + model_name + "_" + topo + "_" + str(NUM_CLIENTS) + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(SEED)
                    # Directory and log file for data distribution logging
                    data_directory = f'./logs/data_inspection/{file_name}'
                    os.makedirs(data_directory, exist_ok=True)
                    data_log_file = os.path.join(data_directory, 'data_distribution.log')
                    data_logger = setup_logger(f'data_logger_{file_name}', data_log_file)
                
                    # Directory and log file for model results logging
                    model_directory = f'./logs/model_result/{file_name}'
                    os.makedirs(model_directory, exist_ok=True)
                    model_log_file = os.path.join(model_directory, 'model_result.log')
                    model_logger = setup_logger('model_logger_{file_name}', model_log_file)
                
                
                    # separate client's dataset: # A list containing all dataloaders
                    if iid:
                        train_subsets = sample_iid_train_data(global_dataset.train_set, NUM_CLIENTS, SIZE, SEED, data_logger)
                        train_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=True, num_workers=12) for i in train_subsets]
                    else:
                        train_subsets = sample_dirichlet_train_data(global_dataset.train_set, NUM_CLIENTS, SIZE * NUM_CLIENTS,
                                                                    ALPHA, SEED, data_logger)
                        train_loaders = [DataLoader(train_subsets[i], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
                                         for i in range(NUM_CLIENTS)]
                
                    # nodes setting
                    nodes_list = create_nodes_list(NUM_CLIENTS, [train_loaders, [0]*NUM_CLIENTS], global_model)
                    if topo != "random":
                        nodes_list = create_adjacency(nodes_list, topo)
                    else:
                        nodes_list = create_custom_adjacency(nodes_list)
                
                    # trainer setting
                    for r in range(ROUND):
                        model_logger.info(f"this is the {r + 1} round.")
                        # training process
                        for client in nodes_list:
                            local_trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", devices="auto", logger=False,
                                                    callbacks = [MyCustomCheckpoint(save_dir=f"saved_models/{file_name}/Local_models/Round_{r+1}",
                                                    idx=client.idx, logger=model_logger)],
                                                    enable_checkpointing=False, enable_model_summary=False)
                            local_trainer.fit(client.model, client.train_loader)
                            client.set_current_params(client.model.state_dict())  # store the current trained model params
                            
                            # local_trainer.test(client.model, client.test_dataloader, verbose=False)
                            # test_result = local_trainer.callback_metrics
                            # merged_results = {**train_result, **test_result}
                            
                    
                        # Aggregation process
                        for client in nodes_list:
                            client.aggregate_weights()
                            
                            save_params(client.nei_agg_params, round_num=r + 1, file_name=file_name, client_id=client.idx,
                                        is_global=False)
                            save_params(client.aggregated_params, round_num=r + 1, file_name=file_name, client_id=client.idx,
                                        is_global=True)


if __name__ == '__main__':
    main()
