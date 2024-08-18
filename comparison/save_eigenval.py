import copy
from robustbench.utils import load_model
import torch
import torch.nn.functional as F
import os
from defenses.riuf import OurModel

names = ['Engstrom2019Robustness', 'Rice2020Overfitting','Carmon2019Unlabeled','Wang2023Better_WRN-28-10']
for model_name in names:
    print("=============",model_name,"=============")
    baseline = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf').cuda()
    our_model = OurModel(baseline, 10, model_name=model_name).cuda()
    our_model.eval()
    del(our_model)
