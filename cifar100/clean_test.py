import copy
import torch
import torch.nn.functional as F
from torch.autograd import grad
from models import *
import torchvision
import torchvision.transforms as transforms
import os
import sys 

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class OurModel(torch.nn.Module):
    def __init__(self, net, eigenvectors, k):
        super(OurModel, self).__init__()
        self.backbone=copy.deepcopy(net)
        self.linear = copy.deepcopy(net.module.linear)
        self.backbone.module.linear= Identity()
        v = eigenvectors[:,:k].cuda() 
        self.eigenbasis = v@v.mT
        print(v.shape)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.matmul(x,self.eigenbasis)
        x = self.linear(x)
        return x


def pgd_attack(model, images, y, epsilon, alpha, num_iters):
    x = images.detach()
    x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for i in range(num_iters):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + alpha * torch.sign(grad.detach())
        x = torch.min(torch.max(x, images - epsilon), images + epsilon)
        x = torch.clamp(x, 0, 1)
    return x

def test_clean(model, device):
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                         std=[0.2023, 0.1994, 0.2010])
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=400, shuffle=False, num_workers=4)
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total

def test_adversarial(model, device, dataloader, eps, alpha, num_iters):
    model.eval()
    correct_fgsm = 0
    correct_pgd = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # PGD Attack
        perturbed_inputs = pgd_attack(model, inputs, targets, eps, alpha, num_iters)
        outputs = model(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_pgd += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct_pgd / total

def compute_truncated_feature_covariance(feature_extractor, trainloader):
    features = []
    for images, labels in trainloader:
        with torch.no_grad():
            images = images.to(device)
            features_batch=feature_extractor(images)
            features.append(features_batch.cpu())
    features = torch.concatenate(features)
    covariance = features.mT @ features
    s, v = torch.linalg.eig(covariance)
    print(features.shape, s.shape, v.shape)
    return features, s.real, v.real

def compute_truncated_robust_forward_model(model, trainloader):
    feature_extractor = copy.deepcopy(model)
    feature_extractor.module.linear = Identity()
    feature_extractor.eval()
    features, s, v = compute_truncated_feature_covariance(feature_extractor, trainloader)
    #torch.save(features, "feature_svd/features.pt")
    #torch.save(s, "feature_svd/s_values.pt")
    #torch.save(v, "feature_svd/v_values.pt")
    return features, s, v

def robust_model_adversarial_eval(net, eigenvectors, device, trainloader, testloader, eps, alpha, num_iters):
    ourmodel = OurModel(net, eigenvectors, 15)
    ourmodel = ourmodel.to(device)
    clean_acc = test_clean(ourmodel, device)
    #test_acc = test_adversarial(ourmodel, device, testloader, eps, alpha, num_iters)
    print(f'Clean: {clean_acc:.2f}')
    return 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing train data')
transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)

print('==> Preparing test data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                         std=[0.2023, 0.1994, 0.2010])
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

# load model
print('==> Resuming from checkpoint..')
checkpoints = ['basic_training', 'pgd_adversarial_training', 'interpolated_adversarial_training', 'basic_training_with_robust_dataset']
basicfile = torch.load('checkpoints/'+checkpoints[int(sys.argv[1])])
net.load_state_dict(basicfile['net'])

print("==> Computing the Train and test robust acc for base model")
base_clean_acc = test_clean(net, device)
#base_pgd_acc =  test_adversarial(net, device, testloader, eps=0.0314, alpha=0.00784, num_iters=40)
print(f'Clean base: {base_clean_acc:.2f}%')
print("==> Computing the eigenvalues and eigenvectors")
_, _, eigenvectors = compute_truncated_robust_forward_model(net, trainloader)
robust_model_adversarial_eval(net, eigenvectors, device, trainloader, testloader, eps=0.0314, alpha=0.00784, num_iters=40)
