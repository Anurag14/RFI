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
testset = torchvision.datasets.CIFAR10(root='../cifar10/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
names = ['Wang2023Better_WRN-28-10']
for model_name in names:
    baseline = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf').cuda()
    start = time.time()
    base_clean_acc = clean_test_acc(baseline, device, testloader)
    end = time.time()
    print((end-start)/10000)



