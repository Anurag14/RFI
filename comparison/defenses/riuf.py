import copy
import torch
from .feature_extractor import FeatureExtractor

def figure_out_beta(model):
    if hasattr(model, 'fc'):
        beta = copy.deepcopy(model.fc.weight) # last layer
    elif hasattr(model, 'logits'):
        beta = copy.deepcopy(model.logits.weight)
    else:
        beta = copy.deepcopy(model.linear.weight)
    return beta
def assign_final_weight(model, final_weight):
    if hasattr(model, 'fc'):
        model.fc.weight = final_weight # last layer
    elif hasattr(model, 'logits'):
        model.logits.weight = final_weight
    else:
        model.linear.weight = final_weight
    return model

def compute_topk_useful(model, eigenvalues, eigenvectors, num_classes):
    beta = figure_out_beta(model)
    eigenvalues, eigenvectors = eigenvalues.cuda(), eigenvectors.cuda()
    ranking = eigenvalues*(beta@eigenvectors)**2
    columns = set()
    for i in range(ranking.shape[0]):
        x=torch.argsort(ranking[i])[-num_classes:].cpu().tolist()
        columns = columns.union(set(x))
    v = eigenvectors[:,sorted(list(columns))]
    model=assign_final_weight(model, torch.nn.Parameter(beta@v@v.T))
    return model

def compute_topk(model, eignvectors, k):
    v = eigenvectors[:,:k].cuda()
    beta = figure_out_beta(model)
    model = assign_final_weight(model, torch.nn.Parameter(beta@v@v.T))
    return model

def OurModel(model, num_features, model_name=None, hard=False):
    _, eigenvalues, eigenvectors = FeatureExtractor(model).compute_feature_covariance()
    if hard:
        model = compute_topk(model, eigenvectors, num_features)
    else:
        model = compute_topk_useful(model, eigenvalues, eigenvectors, num_features)
    return model
