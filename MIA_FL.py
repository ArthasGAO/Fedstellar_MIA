import torch
from openpyxl import load_workbook
from openpyxl import Workbook


class MembershipInferenceAttack:
    def __init__(self, model, global_dataset, in_eval, out_eval, logger, indexing_map):
        self.model = model
        self.global_dataset = global_dataset  # this represents the dataset we use here(MNIST, FMNIST or CIFAR10)

        self.in_eval = in_eval  # evaluation dataloader for in samples: a list containing several dataloaders for each node
        self.out_eval = out_eval  # evaluation dataloader for out samples

        self.logger = logger
        
        self.device = next(self.model.parameters()).device # it should be set before calling any methods that rely on it

        # self.in_eval_pre = [self.compute_predictions(self.model, i) for i in self.in_eval]  
        self.in_eval_pre = self.compute_predictions(self.model, self.in_eval)    # prediction score for in_eval
        self.out_eval_pre = self.compute_predictions(self.model, self.out_eval)  # prediction score for out_eval
        
        self.index_mapping = indexing_map # used to decompose the in_samples for each node

        
        
    def compute_predictions(self, model, dataloader):
        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for inputs, label in dataloader:
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)

                predictions.append(probs)
                labels.append(label)

            predictions = torch.cat(predictions, dim=0) # it has become one tensor
            labels = torch.cat(labels, dim=0) # it has become one tensor
            
        return predictions, labels

    def execute_attack(self):
        # To be overridden by specific attack implementations
        raise NotImplementedError("Must override execute_attack")

    def evaluate_metrics(self, true_p, false_p):
        node_size = len(self.in_eval_pre[0][0])
        size = len(self.out_eval_pre[0])
        
        for idx, item in enumerate(true_p):
            self.logger.info(f"This is the clinet_{idx} result.")
            self.logger.info(f"{item}/{node_size}")
            
        true_p = sum(true_p)

        precision = true_p / (true_p + false_p)
        recall = true_p / size
        fpr = false_p / size
        f1 = 2 * precision * recall / (precision + recall)
        
    def append_experiment_results(self, file_name, data):
        wb = load_workbook(file_name)
        ws = wb.active
        ws.append(data)
        wb.save(file_name)

        