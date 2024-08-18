from robustbench.data import load_cifar100
from robustbench.utils import load_model
from autoattack import AutoAttack
import copy
import torch
import torch.nn.functional as F
from torch.autograd import grad
from models import *
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from robustbench.eval import benchmark

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.backbone = copy.deepcopy(model)
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = Identity()
        else:
            self.backbone.linear = Identity()
    def forward(self, x):
        x=self.backbone(x)
        return x

class OurModel(torch.nn.Module):
    def __init__(self, model, eigenvalues, eigenvectors, num_classes):
        super(OurModel, self).__init__()
        self.backbone = FeatureExtractor(model)
        if hasattr(model, 'fc'):
            self.linear = copy.deepcopy(model.fc) # last layer
        else:
            self.linear = copy.deepcopy(model.linear)
        self.compute_topk(eigenvectors, 70)

    def compute_topk_useful(self, eigenvalues, eigenvectors, num_classes):
        beta = copy.deepcopy(self.linear.weight)
        eigenvalues, eigenvectors = eigenvalues.cuda(), eigenvectors.cuda()
        ranking = (eigenvalues*(beta@eigenvectors))**2
        columns = set()
        for i in range(ranking.shape[0]):
            x=torch.argsort(ranking[i])[-num_classes:].cpu().tolist()
            columns = columns.union(set(x))
        v = eigenvectors[:,sorted(list(columns))]
        print(v.shape, v.T.shape)
        self.linear.weight = torch.nn.Parameter(self.linear.weight@v@v.T)
        return

    def compute_topk(self, eignvectors, k):
        v = eigenvectors[:,:k].cuda()
        self.linear.weight = torch.nn.Parameter(self.linear.weight@v@v.T)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x

## iterate through the cifar 10 dataset to obtain covariance 
def compute_truncated_feature_covariance(feature_extractor):
    print('==> Prepairing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)
    features = []
    print("going inside the loop")
    for i, (images, labels) in enumerate(trainloader):
        with torch.no_grad():
            images = images.cuda()
            features_batch=feature_extractor(images)
            features.append(features_batch)
    print("we are done looping", flush=True)    
    features = torch.concatenate(features)
    print("we are done concatenate", flush=True)
    covariance = features.mT @ features
    print("we made the covariance", flush=True)
    s, v = torch.linalg.eig(covariance)
    print(features.shape, s.shape, v.shape, flush=True)
    return features, s.real, v.real
for model_name in ['Addepalli2022Efficient_RN18']:
    print(model_name)
    model = load_model(model_name, dataset='cifar100', threat_model='Linf').cuda()
    #feature_extractor = FeatureExtractor(model).cuda()
    #features, eigenvalues, eigenvectors = compute_truncated_feature_covariance(feature_extractor)
    #our_model = OurModel(model, eigenvalues, eigenvectors, 10).cuda()
    model.eval()
    #our_model.eval()
    x_test, y_test = load_cifar100(n_examples=10000)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce','apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    #adversary = AutoAttack(our_model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-dlr'])
    #adversary.apgd.n_restarts = 1
    #x_adv = adversary.run_standard_evaluation(x_test, y_test)
