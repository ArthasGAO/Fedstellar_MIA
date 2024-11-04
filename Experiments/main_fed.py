import logging
import os
import sys
from torch.utils.data import DataLoader
from DFL_util import parse_experiment_file, dataset_model_set, create_nodes_list, \
    create_adjacency, sample_iid_train_data, sample_dirichlet_train_data, create_custom_adjacency, \
    save_params
from lightning import Trainer
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
    COMMAND = f'./command/{sys.argv[1]}.txt'

    experiment_details = parse_experiment_file(COMMAND)

    TOPOLOGY = experiment_details["Topology"]
    ROUND = int(experiment_details["Round"])
    NUM_CLIENTS = int(experiment_details["Num_clients"])
    DATASET = experiment_details["Dataset"]
    MODEL = experiment_details["Model"]
    IID = int(experiment_details["iid"])
    BATCH_SIZE = int(experiment_details["batch"])
    SIZE = int(experiment_details["Size"])  # fixed as 2500
    MAX_EPOCHS = int(experiment_details["max_epochs"])  # fixed as 10
    SEED = int(experiment_details["seed"])
    ALPHA = float(experiment_details["alpha"])

    file_name = DATASET + "_" + MODEL + "_" + TOPOLOGY + "_" + str(NUM_CLIENTS) + "_" + str(IID) + "_" + str(ALPHA) + "_" + str(SEED)
    # Directory and log file for data distribution logging
    data_directory = f'./logs/data_inspection/{file_name}'
    os.makedirs(data_directory, exist_ok=True)
    data_log_file = os.path.join(data_directory, 'data_distribution.log')
    data_logger = setup_logger('data_logger', data_log_file)

    # Directory and log file for model results logging
    model_directory = f'./logs/model_result/{file_name}'
    os.makedirs(model_directory, exist_ok=True)
    model_log_file = os.path.join(model_directory, 'model_result.log')
    model_logger = setup_logger('model_logger', model_log_file)

    with open(COMMAND, 'r') as file:
        for line in file:
            model_logger.info(line.strip())

    # dataset and model setting
    global_dataset, global_model = dataset_model_set(DATASET, MODEL, data_logger)

    # separate client's dataset: # A list containing all dataloaders
    if IID:
        train_subsets = sample_iid_train_data(global_dataset.train_set, NUM_CLIENTS, SIZE, SEED, data_logger)
        train_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=True, num_workers=12) for i in train_subsets]
    else:
        train_subsets = sample_dirichlet_train_data(global_dataset.train_set, NUM_CLIENTS, SIZE * NUM_CLIENTS,
                                                    ALPHA, SEED, data_logger)
        train_loaders = [DataLoader(train_subsets[i], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
                         for i in range(NUM_CLIENTS)]

    # nodes setting
    nodes_list = create_nodes_list(NUM_CLIENTS, [train_loaders, [0]*NUM_CLIENTS], global_model)
    if TOPOLOGY != "random":
        nodes_list = create_adjacency(nodes_list, TOPOLOGY)
    else:
        nodes_list = create_custom_adjacency(nodes_list)

    # trainer setting
    for r in range(ROUND):
        model_logger.info(f"this is the {r + 1} round.")
        # training process
        for client in nodes_list:
            local_trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", devices="auto", logger=False,
                                    enable_model_summary=False)
            local_trainer.fit(client.model, client.train_loader)
            # save_model(client.model, round_num=r + 1, time=current_time, client_id=client.idx)
            client.set_current_params(client.model.state_dict())  # store the current trained model params
            train_result = local_trainer.callback_metrics
            # local_trainer.test(client.model, client.test_dataloader, verbose=False)
            # test_result = local_trainer.callback_metrics
            # merged_results = {**train_result, **test_result}
            res_dict = {key: value.item() if hasattr(value, 'item') else value for key, value in train_result.items()}
            model_logger.info(f"{client.idx}'s {r + 1} round model train result: {json.dumps(res_dict, indent=None)}")
            model_logger.info('')
        # Aggregation process
        for client in nodes_list:
            client.aggregate_weights()
            # m_attack = MetricBasedAttack(client.model, 1, train_loaders, test_loaders, model_logger,1)
            # m_attack.MIA_correctness_attack()
            save_params(client.nei_agg_params, round_num=r + 1, file_name=file_name, client_id=client.idx,
                        is_global=False)
            save_params(client.aggregated_params, round_num=r + 1, file_name=file_name, client_id=client.idx,
                        is_global=True)
            # evaluate_model_on_dataloaders(client.model, client.train_dataloader, client.idx, r, model_logger)
            # evaluate_model_on_dataloaders(client.model, client.test_dataloader, client.idx, r, model_logger)


if __name__ == '__main__':
    main()
