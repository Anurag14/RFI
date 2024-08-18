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
    def __init__(self,resnet, k):
        super(OurModel, self).__init__()
        self.backbone=copy.deepcopy(resnet)
        self.linear = copy.deepcopy(resnet.module.linear)
        self.backbone.module.linear= Identity()
        v = torch.load("feature_svd/v_values.pt")[:,:k].cuda() 
        self.eigenbasis = v@v.mT
        print(v.shape)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.matmul(x,self.eigenbasis)
        x = self.linear(x)
        return x

def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, iters=10, restarts=1, device='cuda'):
    """
    PGD L2 attack implementation.
    
    Args:
        model (nn.Module): trained model to attack
        device (torch.device): device to run the attack on
        images (Tensor): batch of input images to attack
        labels (Tensor): corresponding true labels
        epsilon (float): maximum L2 perturbation of the input image
        alpha (float): step size for each iteration of the attack
        iters (int): number of iterations for each restart of the attack
        restarts (int): number of restarts for the attack
    
    Returns:
        perturbed_images (Tensor): batch of perturbed images
    """
    
    # Create a copy of the input images to be perturbed
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    
    # Define the optimization criterion
    criterion = nn.CrossEntropyLoss()
    lower_limit = 0.0
    upper_limit = 1.0
    # Run the attack for the specified number of restarts
    for r in range(restarts):
        
        # Randomly initialize the perturbation
        delta = torch.zeros_like(perturbed_images).uniform_(-epsilon, epsilon)
        delta = torch.clamp(delta, lower_limit - images, upper_limit - images)
        delta = torch.tensor(delta.detach().cpu().numpy(), requires_grad=True).cuda()
        
        # Run the PGD attack for the specified number of iterations
        for i in range(iters):
            
            # Compute the loss and gradient
            logits = model(perturbed_images + delta)
            loss = criterion(logits, labels)
            grad = torch.autograd.grad(loss, delta)[0]
            
            # Compute the L2 norm of the gradient
            l2_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            
            # Normalize the gradient by the L2 norm
            grad_norm = grad / (l2_norm + 1e-10)
            
            # Update the perturbation using the normalized gradient
            delta.data.add_(alpha * grad_norm.data)
            
            # Project the perturbation onto the L2 ball
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
            delta.data.clamp_(lower_limit - perturbed_images.data, upper_limit - perturbed_images.data)
            
            # Clamp the perturbed image to the valid range
            perturbed_images.data.clamp_(lower_limit, upper_limit)
        
        # Choose the perturbed image with the largest loss
        logits = model(perturbed_images + delta)
        loss = criterion(logits, labels)
        if r == 0:
            max_loss = loss.clone().detach()
            best_delta = delta.clone().detach()
        else:
            replace = loss > max_loss
            max_loss[replace] = loss[replace]
            best_delta[replace] = delta[replace].clone().detach()
        
        # Reset the perturbed image and perturbation for the next restart
        perturbed_images = images.clone().detach()
        delta = torch.zeros_like(perturbed_images).uniform_(-epsilon, epsilon)
        delta = torch.clamp(delta, lower_limit - images, upper_limit - images)
        delta = torch.tensor(delta.detach().cpu().numpy(), requires_grad=True).cuda()
    
    # Add the best perturbation to the original image and return
    perturbed_images = torch.clamp(perturbed_images + best_delta, lower_limit, upper_limit)
    return perturbed_images


def test_adversarial_adapted(ourmodel, resnet, device, testloader, eps, alpha, num_iters):
    ourmodel.eval()
    resnet.eval()
    correct_pgd = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # PGD Attack
        perturbed_inputs = pgd_attack(ourmodel, inputs, targets, eps, alpha, num_iters)
        outputs = ourmodel(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_pgd += predicted.eq(targets).sum().item()
        total += targets.size(0)
    print(f'PGD Attack Accuracy: {100.0 * correct_pgd / total:.2f}%')

def test_adversarial(model, device, testloader, eps, alpha, num_iters):
    model.eval()
    correct_pgd = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # PGD Attack
        perturbed_inputs = pgd_attack(model, inputs, targets, eps, alpha, num_iters)
        outputs = model(perturbed_inputs)
        _, predicted = outputs.max(1)
        correct_pgd += predicted.eq(targets).sum().item()
        total += targets.size(0)
    print(f'PGD Attack Accuracy: {100.0 * correct_pgd / total:.2f}%')

def compute_truncated_feature_covariance(feature_extractor):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
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
    return features, s.real, v.real

def compute_truncated_robust_forward_model(model):
    feature_extractor = copy.deepcopy(model)
    feature_extractor.module.linear = Identity()
    feature_extractor.eval()
    #print("feature extractor: ",feature_extractor)
    features, s, v = compute_truncated_feature_covariance(feature_extractor)
    torch.save(features, "feature_svd/features.pt")
    torch.save(s, "feature_svd/s_values.pt")
    torch.save(v, "feature_svd/v_values.pt")
    return 

def robust_model_adversarial_eval(resnet, device, testloader, eps, alpha, num_iters):
    for k in range(1,20):
        print("Running with top-"+str(k)+" features")
        ourmodel = OurModel(resnet, k)
        ourmodel = ourmodel.to(device)
        test_adversarial_adapted(ourmodel, resnet, device, testloader, eps, alpha, num_iters)
    return 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                         std=[0.2023, 0.1994, 0.2010])
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
checkpoints = ['basic_training', 'pgd_adversarial_training', 'interpolated_adversarial_training', 'basic_training_with_robust_dataset']
basicfile = torch.load('checkpoints/'+checkpoints[1])
net.load_state_dict(basicfile['net'])
test_adversarial(net, device, testloader, eps=0.5, alpha=0.1, num_iters=100)
compute_truncated_robust_forward_model(net)
robust_model_adversarial_eval(net, device, testloader, eps=0.5, alpha=0.1, num_iters=100)
