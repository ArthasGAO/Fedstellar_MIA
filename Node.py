import copy
import torch


class Node:
    def __init__(self, idx, model, train_loader, test_loader):
        self.idx = idx
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.neigh = set()
        self.current_params = None
        self.nei_agg_params = None

    def aggregate_weights(self):
        if not self.neigh:
            # If there are no neighbors, we might want to skip the update
            return

        # Initialize the aggregated weights with the current node's parameters
        aggregated_weights = {k: torch.zeros_like(v) for k, v in self.current_params.items()}
        nei_aggregated_weights = {k: torch.zeros_like(v) for k, v in self.current_params.items()}

        # Accumulate the weights from the neighbors
        for node in self.neigh:
            node_weights = node.current_params
            for k in aggregated_weights.keys():
                aggregated_weights[k] += node_weights[k]
            for k in nei_aggregated_weights.keys():
                nei_aggregated_weights[k] += node_weights[k]

        # Average the weights (including the current node's weights)
        num_nodes = len(self.neigh) + 1
        for k in aggregated_weights.keys():
            aggregated_weights[k] += self.current_params[k]
            aggregated_weights[k] /= num_nodes

        # Average the weights (excluding the current node's weights)
        for k in nei_aggregated_weights.keys():
            nei_aggregated_weights[k] /= len(self.neigh)

        # Apply the aggregated weights to the model
        with torch.no_grad():  # Ensure gradients are not tracked for this operation
            self.model.load_state_dict(aggregated_weights, strict=False)

        self.nei_agg_params = nei_aggregated_weights
        self.aggregated_params = aggregated_weights
        
    def set_current_params(self, params):
        self.current_params = copy.deepcopy(params)
        
    def get_current_params(self):
        return copy.deepcopy(self.current_params)
