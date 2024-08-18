import copy
import torch
import torch.nn.functional as F
from torch.autograd import grad
from models import *
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class OurModel(torch.nn.Module):
    def __init__(self,resnet):
        super(OurModel, self).__init__()
        self.backbone=copy.deepcopy(resnet)
        self.linear = copy.deepcopy(resnet.module.linear)
        self.backbone.module.linear= Identity()
        v = torch.load("feature_svd/v_values.pt")[:,:10].cuda() 
        self.eigenbasis = v@v.mT
        print(v.shape)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.matmul(x,self.eigenbasis)
        x = self.linear(x)
        return x

def fgsm_attack(model, images, labels, eps):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    gradients = grad(loss, images)[0]
    signed_gradients = gradients.sign()
    perturbed_images = images + eps * signed_gradients
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

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

def test_adversarial_adapted(ourmodel, resnet, device, testloader, eps, alpha, num_iters):
    ourmodel.eval()
    resnet.eval()
    correct_fgsm = 0
    correct_pgd = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # FGSM Attack
        perturbed_inputs = fgsm_attack(resnet, inputs, targets, eps)
        outputs = ourmodel(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_fgsm += predicted.eq(targets).sum().item()
        # PGD Attack
        perturbed_inputs = pgd_attack(resnet, inputs, targets, eps, alpha, num_iters)
        outputs = ourmodel(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_pgd += predicted.eq(targets).sum().item()
        total += targets.size(0)
    print(f'FGSM Attack Accuracy: {100.0 * correct_fgsm / total:.2f}%')
    print(f'PGD Attack Accuracy: {100.0 * correct_pgd / total:.2f}%')

def test_adversarial(model, device, testloader, eps, alpha, num_iters):
    model.eval()
    correct_fgsm = 0
    correct_pgd = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # FGSM Attack
        perturbed_inputs = fgsm_attack(model, inputs, targets, eps)
        outputs = model(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_fgsm += predicted.eq(targets).sum().item()
        # PGD Attack
        perturbed_inputs = pgd_attack(model, inputs, targets, eps, alpha, num_iters)
        outputs = model(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_pgd += predicted.eq(targets).sum().item()
        total += targets.size(0)
    print(f'FGSM Attack Accuracy: {100.0 * correct_fgsm / total:.2f}%')
    print(f'PGD Attack Accuracy: {100.0 * correct_pgd / total:.2f}%')

def compute_truncated_feature_covariance(feature_extractor):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)
    features = []
    for idx, (images, labels) in tqdm(enumerate(trainloader)):
        with torch.no_grad():
            images = images.to(device)
            features_batch=feature_extractor(images)
            features.append(features_batch.cpu())
    features = torch.concatenate(features)
    covariance = features.mT @ features
    s, v = torch.linalg.eig(covariance)
    print(features.shape, s.shape, v.shape)
    return s.real, v.real

def compute_truncated_robust_forward_model(model):
    model.module.linear = Identity()
    feature_extractor = model  
    feature_extractor.eval()
    print("feature extractor: ",feature_extractor)
    s, v = compute_truncated_feature_covariance(feature_extractor)
    torch.save(s, "feature_svd/s_values.pt")
    torch.save(v, "feature_svd/v_values.pt")
    return 

def robust_model_adversarial_eval(resnet, device, testloader, eps, alpha, num_iters):
    ourmodel = OurModel(resnet)
    ourmodel = ourmodel.to(device)
    test_adversarial_adapted(ourmodel, resnet, device, testloader, eps, alpha, num_iters)
    return 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

# load model
print('==> Resuming from checkpoint..')
basicfile = torch.load('basic_training')
net.load_state_dict(basicfile['net'])
compute_truncated_robust_forward_model(net)
robust_model_adversarial_eval(net, device, testloader, eps=0.0314, alpha=0.00784, num_iters=7)
#test_adversarial(net, device, testloader, eps=0.0314, alpha=0.00784, num_iters=7)
