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
    def __init__(self, net, eigenvalues, eigenvectors, num_classes):
        super(OurModel, self).__init__()
        self.backbone = copy.deepcopy(net)
        if sys.argv[1] in ['4','5']:
            self.linear = copy.deepcopy(net.linear)
            self.backbone.linear = Identity()
        else:
            self.linear = copy.deepcopy(net.module.linear)
            self.backbone.module.linear = Identity()
        
        #self.compute_topk_useful(eigenvalues, eigenvectors, num_classes)
        #self.compute_topk_useful_new(eigenvalues, eigenvectors, num_classes)
        self.compute_topk(eigenvectors, 100)

    def compute_topk_classwise(self, eigenvalues, eigenvectors, num_classes):
        beta = copy.deepcopy(self.linear.weight)
        ranking = (eigenvalues*(beta.cpu()@eigenvectors))**2
        print(ranking)
        v = torch.stack([eigenvectors[:,sorted(torch.argsort(ranking[i])[-num_classes:].tolist())] for i in range(num_classes)])
        v_vt = torch.bmm(v, torch.transpose(v, 1,2))
        self.linear.weight = torch.nn.Parameter(torch.bmm(self.linear.weight[:,None,:], v_vt).squeeze(1))
        return 
    
    def compute_topk_useful(self, eigenvalues, eigenvectors, num_classes):
        beta = copy.deepcopy(self.linear.weight)
        eigenvalues, eigenvectors = eigenvalues.cuda(), eigenvectors.cuda()
        ranking = (eigenvalues*(beta@eigenvectors))**2
        columns = set()
        for i in range(ranking.shape[0]):
            x=torch.argsort(ranking[i])[-num_classes:].cpu().tolist()
            print(x)
            columns = columns.union(set(x))     
        print(columns)
        v = eigenvectors[:,sorted(list(columns))]
        print(v.shape, v.T.shape)
        self.linear.weight = torch.nn.Parameter(self.linear.weight@v@v.T)
        return 
    
    def compute_topk_useful_new(self, eigenvalues, eigenvectors, num_classes):
        beta = copy.deepcopy(self.linear.weight)
        eigenvalues, eigenvectors = eigenvalues.cuda(), eigenvectors.cuda()
        ranking = eigenvalues*(beta@eigenvectors)**2
        columns = set()
        for i in range(ranking.shape[0]):
            x=torch.argsort(ranking[i])[-num_classes:].cpu().tolist()
            print(x)
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

def test_adversarial_transfer(model, resnet, device, dataloader, eps, alpha, num_iters):
    model.eval()
    resnet.eval()
    correct_fgsm = 0
    correct_pgd = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        perturbed_inputs = pgd_attack(resnet, inputs, targets, eps, alpha, num_iters)
        outputs = model(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_pgd += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct_pgd / total

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
    if sys.argv[1] in ['4','5']:
        feature_extractor.linear = Identity()
    else:
        feature_extractor.module.linear = Identity()
    feature_extractor.eval()
    features, s, v = compute_truncated_feature_covariance(feature_extractor, trainloader)
    return features, s.cpu(), v.cpu()

def robust_model_adversarial_eval(net, eigenvalues, eigenvectors, device, testloader, eps, alpha, num_iters):
    ourmodel = OurModel(net, eigenvalues, eigenvectors, 10)
    ourmodel = ourmodel.to(device)
    test_acc = test_adversarial(ourmodel, device, testloader, eps, alpha, num_iters)
    
    transfer_acc = test_adversarial_transfer(ourmodel, net, device, testloader, eps, alpha, num_iters)
    clean_acc = clean_test_acc(ourmodel, device, testloader)
    
    print(f'Clean (Ours): {clean_acc:.2f}% PGD (Test): {test_acc:.2f}% PGD (Transfer): {transfer_acc:.2f}%')
    return 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing train data')
transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

print('==> Preparing test data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

net = ResNet18(100)
net = net.to(device)

# load model
print('==> Resuming from checkpoint..')
checkpoints = ['basic_training', 
        'pgd_adversarial_training', 
        'interpolated_adversarial_training', 
        'basic_training_with_robust_dataset',
        'carlini_training',
        'trades.pt']
if int(sys.argv[1]) == 4:
    net = torch.nn.DataParallel(net)
    net = torch.load('checkpoints/'+checkpoints[int(sys.argv[1])])
elif int(sys.argv[1]) == 5:
    basicfile = torch.load('checkpoints/'+checkpoints[int(sys.argv[1])])
    net.load_state_dict(basicfile)
else:
    net = torch.nn.DataParallel(net)
    basicfile = torch.load('checkpoints/'+checkpoints[int(sys.argv[1])])
    net.load_state_dict(basicfile['net'])
#print(net)
print("==> Computing the Train and test robust acc for base model")
base_clean_acc = clean_test_acc(net, device, testloader)
eps, num_iters = 8/255, 40
base_test_acc =  test_adversarial(net, device, testloader, eps=eps, alpha=eps/4, num_iters=num_iters)
print(f'Clean(Base): {base_clean_acc:.2f}% PGD (Base Test): {base_test_acc:.2f}%')
print("==> Computing the eigenvalues and eigenvectors")
_, eigenvalues, eigenvectors = compute_truncated_robust_forward_model(net, trainloader)
robust_model_adversarial_eval(net, eigenvalues, eigenvectors, device, testloader, eps=eps, alpha=eps/4, num_iters=num_iters)

