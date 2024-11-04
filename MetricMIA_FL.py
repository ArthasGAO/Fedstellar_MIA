import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from PIL import Image
from sklearn.cluster import SpectralClustering
from MIA_FL import MembershipInferenceAttack
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation, CIFAR10DatasetExtendedAugmentation


class MetricBasedAttack(MembershipInferenceAttack):
    def __init__(self, model, global_dataset, in_eval, out_eval, logger, index_mapping, train_result):
        super().__init__(model, global_dataset, in_eval, out_eval, logger, index_mapping)
        self.train_result = train_result

    def execute_attack(self):
        # the method to perform MIAs defined in this class
        for attr_name in dir(self):
            # Check if the attribute is a method and starts with "MIA"
            if attr_name.startswith("MIA") and callable(getattr(self, attr_name)):
                method = getattr(self, attr_name)
                method()

    def MIA_correctness_attack(self):

        def correctness_check(dataset):
            predictions, labels = dataset

            _, predicted_labels = torch.max(predictions, dim=1)

            correct_predictions = predicted_labels == labels

            # true_items = correct_predictions.sum().item()

            return correct_predictions
            
        tp = []    
        
        correct_predictions = correctness_check(self.in_eval_pre)
        for start_idx, end_idx in self.index_mapping:
            node_correct_predictions = correct_predictions[start_idx:end_idx]
            true_positives = node_correct_predictions.sum().item()
            tp.append(true_positives)
        
            
        false_correct_predictions = correctness_check(self.out_eval_pre)
        false_positives = false_correct_predictions.sum().item()
        tp.append(false_positives)
        
        self.append_experiment_results(self.logger["table_dir"], ['PC MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + tp)



    def MIA_loss_attack(self):
        loss_threshold = self.train_result
        
        in_losses = []; out_losses = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.in_eval:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                logits = self.model(inputs)
                losses = F.cross_entropy(logits, labels, reduction='none')
                # true_positives += (losses < loss_threshold).sum().item()
                in_losses.append(losses)

            for inputs, labels in self.out_eval:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                logits = self.model(inputs)
                losses = F.cross_entropy(logits, labels, reduction='none')
                # false_positives += (losses < loss_threshold).sum().item()
                out_losses.append(losses)
        
        tp = []      
        for start_idx, end_idx in self.index_mapping:
            node_losses = in_losses[start_idx:end_idx]
            true_positives = (node_losses < loss_threshold).sum().item()
            tp.append(true_positives)
            
        false_positives = (out_losses < loss_threshold).sum().item() # we treat data with loss smaller than threshold as the in_samples; vice versa
        tp.append(false_positives)
        
        self.append_experiment_results(self.logger["table_dir"], ['PL MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + tp)

    def _generate_random_images(self, batch_size):
        images = []
        
        # np.random.seed(42)
        
        if isinstance(self.global_dataset, CIFAR10Dataset):
            height, width, channels = 32, 32, 3
            mean, std = [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]
            
            if isinstance(self.global_dataset, CIFAR10DatasetNoAugmentation):
                transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
            elif isinstance(self.global_dataset, CIFAR10DatasetExtendedAugmentation):
                transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
            else:
                transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
        else:  # gray scale images
            height, width, channels = 28, 28, 1
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            
        # Generate random images
        for _ in range(batch_size):
            data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            img = Image.fromarray(data.squeeze() if channels == 1 else data)
            images.append(img)
        
        # Apply transformations
        transformed_images = [transform(img) for img in images]

        return torch.stack(transformed_images)

    def _compute_entropy(self, probs):
        log_probs = torch.log(probs + 1e-6)  # Correctly use log on probabilities
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy

    def _threshold_choosing(self, m_name):
        random_images = self._generate_random_images(batch_size=len(self.out_eval_pre[0]))
        random_dataloader = DataLoader(TensorDataset(random_images), batch_size=128, shuffle=False, num_workers=12)

        threshold = []

        self.model.eval()
        with torch.no_grad():
            for batch in random_dataloader:
                inputs = batch[0].to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                if m_name == "confidence":
                    confidences, _ = torch.max(probs, dim=1)
                    threshold.append(confidences)
                else:
                    entropies = self._compute_entropy(probs)
                    threshold.append(entropies)

        threshold_tensor = torch.cat(threshold)

        sequence = list(range(10, 100, 10)) + [95]
        threshold_percentiles = [np.percentile(threshold_tensor.cpu().detach().numpy(), i) for i in sequence]

        return threshold_percentiles # it contains 10 percentiles as the backup thresholds

    def MIA_maximal_confidence_attack(self):
        threshold = self._threshold_choosing("confidence")

        def maximal_confidence_check(dataset):
            predictions, labels = dataset

            confidences, _ = torch.max(predictions, dim=1)

            # true_items = (confidences >= thre).sum().item()

            return confidences
            
        best_f1 = 0
        p_ls = []
        r_ls = []
        
        for i, thre in enumerate(threshold):
            tp_list = []
            
            in_confidences = maximal_confidence_check(self.in_eval_pre)
            
            tp = []; true_positives = 0  
            for start_idx, end_idx in self.index_mapping:
                node_maximal_confidences = in_confidences[start_idx:end_idx]
                node_tp = (node_maximal_confidences >= thre).sum().item()
                tp.append(node_tp)
                true_positives += node_tp
            
            out_confidences = maximal_confidence_check(self.out_eval_pre)
            false_positives = (out_confidences >= thre).sum().item()
            
            total_positives = true_positives + false_positives  # This is the denominator for precision
            
            precision = true_positives / total_positives if total_positives > 0 else 0
            recall = true_positives / 25000 
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            p_ls.append(precision)
            r_ls.append(recall)

            # Update best threshold based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_tp_list = tp
                best_fp = false_positives
                
        best_tp_list.append(best_fp)
        
        self.append_experiment_results(self.logger["table_dir"], ['PMC MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + best_tp_list) 
                                                                   
        self.append_experiment_results(self.logger["random_thre_dir"], ['PMC MIA', self.logger["Adversary"], self.logger["topo"],self.logger["iid"],
                                                                        self.logger["Round"], self.logger["Node"]]+threshold+p_ls+r_ls)


    def MIA_entropy_attack(self):
        threshold = self._threshold_choosing("entropy")

        def entropy_check(dataset):
            predictions, labels = dataset

            entropies = self._compute_entropy(predictions)

            # true_items = (entropies <= thre).sum().item()

            return entropies
            
        best_f1 = 0
        
        p_ls = []
        r_ls = []
        
        for i, thre in enumerate(threshold):
            tp_list = []
            
            in_entropies = entropy_check(self.in_eval_pre)
            
            tp = []; true_positives = 0  
            for start_idx, end_idx in self.index_mapping:
                node_entropies = in_entropies[start_idx:end_idx]
                node_tp = (node_entropies <= thre).sum().item()
                tp.append(node_tp)
                true_positives += node_tp
            
            out_entropies = entropy_check(self.out_eval_pre)
            false_positives = (out_entropies <= thre).sum().item()
            
            total_positives = true_positives + false_positives  # This is the denominator for precision
            
            precision = true_positives / total_positives if total_positives > 0 else 0
            recall = true_positives / 25000
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            p_ls.append(precision)
            r_ls.append(recall)

            # Update best threshold based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_tp_list = tp
                best_fp = false_positives
                
        best_tp_list.append(best_fp)
        
        self.append_experiment_results(self.logger["table_dir"], ['PE MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + best_tp_list)
                                                                  
        self.append_experiment_results(self.logger["random_thre_dir"], ['PE MIA', self.logger["Adversary"], self.logger["topo"],self.logger["iid"],
                                                                        self.logger["Round"], self.logger["Node"]]+threshold+p_ls+r_ls)
                                                                  
    def compute_jacobian_and_norm(model, inputs):
        # Move inputs to the GPU
        inputs = inputs.to(self.device)
        # Ensure input tensor requires gradient
        inputs.requires_grad_(True)
        
        # Compute the Jacobian for the current sample
        jacobian_matrix = jacobian(lambda x: model(x), inputs)
        
        # Reshape and store the Jacobian matrix
        jacobian_reshaped = jacobian_matrix.squeeze().reshape(inputs.size(1), -1)  # Reshape to 2D
        l2_norm = torch.norm(jacobian_reshaped, p=2)
        return l2_norm.item()
                                                                  
    def sensitivity_mia(self):
        norms = []
        
        self.model.eval()
        
        # Compute norms for in_eval_small_batch
        for inputs, _ in in_eval:
            l2_norm = compute_jacobian_and_norm(self.model, inputs)
            norms.append(l2_norm)
        
        # Compute norms for out_eval_small_batch
        for inputs, _ in out_eval:
            l2_norm = compute_jacobian_and_norm(self.model, inputs)
            norms.append(l2_norm)
        
        norm_array = np.array(norms)
        
        attack_cluster = SpectralClustering(n_clusters=6, n_jobs=-1, affinity='nearest_neighbors', n_neighbors=19)
        y_attack_pred = attack_cluster.fit_predict(norm_array.reshape(-1, 1))
        split = 1
        
        cluster_1 = np.where(y_attack_pred >= split)[0]
        cluster_0 = np.where(y_attack_pred < split)[0]
        
        y_attack_pred[cluster_1] = 1
        y_attack_pred[cluster_0] = 0
        cluster_1_mean_norm = norm_array[cluster_1].mean()
        cluster_0_mean_norm = norm_array[cluster_0].mean()
        if cluster_1_mean_norm > cluster_0_mean_norm:
            y_attack_pred = np.abs(y_attack_pred-1)
            
        size = 25000
        
        tp = [];  
        for start_idx, end_idx in self.index_mapping:
            node_norm_label = y_attack_pred[start_idx:end_idx]
            node_tp = (node_norm_label == 1).sum().item()
            tp.append(node_tp)
            
            
        limit = self.index_mapping[-1][1]
        
        false_positives = 0
        for i in range(limit, limit+size):
            if y_attack_pred[i] == 1:
              false_positives += 1
              
        tp.append(false_positives)
        
        
        self.append_experiment_results(self.logger["table_dir"], ['PS MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + tp)
            
        
        
        

        

