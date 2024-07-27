import logging
from typing import OrderedDict, List, Optional
import copy
from typing import Optional
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import OrderedDict, List


def cosine_metric2(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        logging.info("Cosine similarity cannot be computed due to missing model")
        return None

    cos_similarities = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()
            if l1.shape != l2.shape:
                # Adjust the shape of the smaller layer to match the larger layer
                min_len = min(l1.shape[0], l2.shape[0])
                l1, l2 = l1[:min_len], l2[:min_len]

            cos_sim = torch.nn.functional.cosine_similarity(l1.unsqueeze(0), l2.unsqueeze(0), dim=1)
            cos_similarities.append(cos_sim.item())

    if cos_similarities:
        avg_cos_sim = torch.mean(torch.tensor(cos_similarities))
        # result = torch.clamp(avg_cos_sim, min=0).item()
        # return result
        return avg_cos_sim.item() if similarity else (1 - avg_cos_sim.item())
    else:
        return None
    
def cosine_metric(model1: OrderedDict, model2: OrderedDict, similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        logging.info("Cosine similarity cannot be computed due to missing model")
        return None

    cos_similarities: List = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].to('cpu')
            l2 = model2[layer].to('cpu')
            if l1.shape != l2.shape:
                # Adjust the shape of the smaller layer to match the larger layer
                min_len = min(l1.shape[0], l2.shape[0])
                l1, l2 = l1[:min_len], l2[:min_len]
            cos = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
            cos_mean = torch.mean(cos(l1.float(), l2.float())).mean()
            cos_similarities.append(cos_mean)
        else:
            logging.info("Layer {} not found in model 2".format(layer))

    if cos_similarities:    
        cos = torch.Tensor(cos_similarities)
        avg_cos = torch.mean(cos)
        relu_cos = torch.nn.functional.relu(avg_cos)  # relu to avoid negative values
        return relu_cos.item() if similarity else (1 - relu_cos.item())
    else:
        return None
        

def euclidean_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], standardized: bool = False, similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()
            if standardized:
                l1 = (l1 - l1.mean()) / l1.std()
                l2 = (l2 - l2.mean()) / l2.std()
            
            distance = torch.norm(l1 - l2, p=2)
            if similarity:
                norm_sum = torch.norm(l1, p=2) + torch.norm(l2, p=2)
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None
    

def minkowski_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], p: int, similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()

            distance = torch.norm(l1 - l2, p=p)
            if similarity:
                norm_sum = torch.norm(l1, p=p) + torch.norm(l2, p=p)
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None


def manhattan_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    distances = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()

            distance = torch.norm(l1 - l2, p=1)
            if similarity:
                norm_sum = torch.norm(l1, p=1) + torch.norm(l2, p=1)
                similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                distances.append(similarity_score.item())
            else:
                distances.append(distance.item())

    if distances:
        avg_distance = torch.mean(torch.tensor(distances))
        return avg_distance.item()
    else:
        return None


def pearson_correlation_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    correlations = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()

            if l1.shape != l2.shape:
                min_len = min(l1.shape[0], l2.shape[0])
                l1, l2 = l1[:min_len], l2[:min_len]

            correlation = torch.corrcoef(torch.stack((l1, l2)))[0, 1]
            if similarity:
                adjusted_similarity = (correlation + 1) / 2
                correlations.append(adjusted_similarity.item())
            else:
                correlations.append(1 - (correlation + 1) / 2)

    if correlations:
        avg_correlation = torch.mean(torch.tensor(correlations))
        return avg_correlation.item()
    else:
        return None


def jaccard_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], similarity: bool = False) -> Optional[float]:
    if model1 is None or model2 is None:
        return None

    jaccard_scores = []

    for layer in model1:
        if layer in model2:
            l1 = model1[layer].flatten()
            l2 = model2[layer].flatten()

            intersection = torch.sum(torch.min(l1, l2))
            union = torch.sum(torch.max(l1, l2))

            jaccard_sim = intersection / union if union != 0 else 0
            if similarity:
                jaccard_scores.append(jaccard_sim.item())
            else:
                jaccard_scores.append(1 - jaccard_sim.item())

    if jaccard_scores:
        avg_jaccard = torch.mean(torch.tensor(jaccard_scores))
        return avg_jaccard.item()
    else:
        return None


def normalise_layers(untrusted_params, trusted_params):
    trusted_norms = dict([k, torch.norm(trusted_params[k].data.view(-1).float())] for k in trusted_params.keys())

    normalised_params = copy.deepcopy(untrusted_params)

    state_dict = copy.deepcopy(untrusted_params)
    for layer in untrusted_params:
        layer_norm = torch.norm(state_dict[layer].data.view(-1).float())
        scaling_factor = min(layer_norm / trusted_norms[layer], 1)
        logging.debug("[Aggregator.normalise_layers()] Layer: {} ScalingFactor {}".format(layer, scaling_factor))
        # logging.info("Scaling client {} layer {} with factor {}".format(client, layer, scaling_factor))
        normalised_layer = torch.mul(state_dict[layer], scaling_factor)
        normalised_params[layer] = normalised_layer

    return normalised_params