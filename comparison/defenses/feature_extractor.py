import torch
import torchvision
import torchvision.transforms as transforms
import copy 

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.backbone = copy.deepcopy(model)
        if hasattr(model, 'fc'):
            self.backbone.fc = Identity()
        elif hasattr(model, 'logits'):
            self.backbone.logits = Identity()
        else:
            self.backbone.linear = Identity()
    def forward(self, x):
        x=self.backbone(x)
        return x

    def compute_feature_covariance(self):
        print('==> Prepairing data..')
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=False, num_workers=4)
        features = []
        for i, (images, _) in enumerate(trainloader):
            with torch.no_grad():
                images = images.cuda()
                features_batch=self.backbone(images)
                features.append(features_batch)
        features = torch.concatenate(features)
        covariance = features.mT @ features
        s, v = torch.linalg.eig(covariance)
        return features, s.real, v.real
