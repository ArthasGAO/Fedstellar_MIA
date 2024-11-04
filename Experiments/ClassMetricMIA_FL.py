import numpy as np
import torch
from MIA_FL import MembershipInferenceAttack


class ClassMetricBasedAttack(MembershipInferenceAttack):
    def __init__(self, model,  global_dataset, in_eval, out_eval, logger, indexing_map, shadow_in_eval, shadow_out_eval):
        super().__init__(model, global_dataset, in_eval, out_eval, logger, indexing_map)

        self.num_classes = 10
        
        self.logger = logger
  
        self.s_in_outputs, self.s_in_labels = shadow_in_eval # Unpack shadow_in_eval and shadow_out_eval
        self.s_out_outputs, self.s_out_labels = shadow_out_eval
        
        self.t_in_outputs, self.t_in_labels = self.in_eval_pre # Unpack in_eval_pre and out_eval_pre
        self.t_out_outputs, self.t_out_labels = self.out_eval_pre
        
        # Move tensors to CPU before converting to NumPy
        self.s_in_outputs = self.s_in_outputs.cpu().detach().numpy()
        self.s_in_labels = self.s_in_labels.cpu().detach().numpy()
        self.s_out_outputs = self.s_out_outputs.cpu().detach().numpy()
        self.s_out_labels = self.s_out_labels.cpu().detach().numpy()
        self.t_in_outputs = self.t_in_outputs.cpu().detach().numpy()
        self.t_in_labels = self.t_in_labels.cpu().detach().numpy()
        self.t_out_outputs = self.t_out_outputs.cpu().detach().numpy()
        self.t_out_labels = self.t_out_labels.cpu().detach().numpy()

        self.s_in_conf = np.array([self.s_in_outputs[i, self.s_in_labels[i]] for i in range(len(self.s_in_labels))])
        self.s_out_conf = np.array([self.s_out_outputs[i, self.s_out_labels[i]] for i in range(len(self.s_out_labels))])
        self.t_in_conf = np.array([self.t_in_outputs[i, self.t_in_labels[i]] for i in range(len(self.t_in_labels))])
        self.t_out_conf = np.array([self.t_out_outputs[i, self.t_out_labels[i]] for i in range(len(self.t_out_labels))])

        self.s_in_entr = self._entr_comp(self.s_in_outputs)
        self.s_out_entr = self._entr_comp(self.s_out_outputs)
        self.t_in_entr = self._entr_comp(self.t_in_outputs)
        self.t_out_entr = self._entr_comp(self.t_out_outputs)

        self.s_in_m_entr = self._m_entr_comp(self.s_in_outputs, self.s_in_labels)
        self.s_out_m_entr = self._m_entr_comp(self.s_out_outputs, self.s_out_labels)
        self.t_in_m_entr = self._m_entr_comp(self.t_in_outputs, self.t_in_labels)
        self.t_out_m_entr = self._m_entr_comp(self.t_out_outputs, self.t_out_labels)
        
        
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre(self, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        
        thresholds = {}
        for num in range(self.num_classes):
            thresholds[num] = self._thre_setting(s_tr_values[self.s_in_labels == num], s_te_values[self.s_out_labels == num])
            
        tp = []
        
        for start_idx, end_idx in self.index_mapping:
            node_t_tr_values = t_tr_values[start_idx:end_idx]
            node_t_in_labels = self.t_in_labels[start_idx:end_idx]
            
            true_positives = 0
            for num in range(self.num_classes):
                threshold = thresholds[num]
                true_positives += np.sum(node_t_tr_values[node_t_in_labels == num] >= threshold)
                # false_positives += np.sum(node_t_te_values[node_t_out_labels == num] >= threshold)
            
            tp.append(true_positives)
        
        false_positives = 0
        for num in range(self.num_classes):
            threshold = thresholds[num]
            false_positives += np.sum(t_te_values[self.t_out_labels == num] >= threshold)
        
        tp.append(false_positives)
        
        return tp

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if all_methods or ('confidence' in benchmark_methods):
            tp = self._mem_inf_thre(self.s_in_conf, self.s_out_conf, self.t_in_conf, self.t_out_conf)
            self.append_experiment_results(self.logger["table_dir"], ['CLC MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + tp)
        if all_methods or ('entropy' in benchmark_methods):
            tp = self._mem_inf_thre(-self.s_in_entr, -self.s_out_entr, -self.t_in_entr, -self.t_out_entr)
            self.append_experiment_results(self.logger["table_dir"], ['CLE MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + tp)
        if all_methods or ('modified entropy' in benchmark_methods):
            tp = self._mem_inf_thre(-self.s_in_m_entr, -self.s_out_m_entr, -self.t_in_m_entr, -self.t_out_m_entr)
            self.append_experiment_results(self.logger["table_dir"], ['MCLE MIA', self.logger["Adversary"], self.logger["topo"], self.logger["iid"],
                                                                  self.logger["Round"], self.logger["Node"]] + tp)


