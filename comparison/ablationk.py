import time 
import copy
from robustbench.utils import load_model
import sys
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
import os
from defenses.riuf import OurModel

def clean_test_acc(model, device, dataloader):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing test data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR100(root='../cifar100/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=8)
names = ['Addepalli2022Efficient_RN18','Rice2020Overfitting','Pang2022Robustness_WRN28_10','Wang2023Better_WRN-28-10']
ks=range(0,100,10)
for model_name in names:
    print("=============",model_name,"=============")
    ret={}
    for k in ks:
        baseline = load_model(model_name=model_name, dataset='cifar100', threat_model='Linf').cuda()
        our_model = OurModel(baseline, k).cuda()
        our_model.eval()
        base_clean_acc = clean_test_acc(our_model, device, testloader)
        print(base_clean_acc)
        ret[k]=base_clean_acc
        del(our_model)
    print(ret)
        

