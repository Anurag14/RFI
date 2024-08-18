import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchdiffeq import odeint_adjoint as odeint
from .feature_extractor import FeatureExtractor 
import torchvision
import torchvision.transforms as transforms

class ConcatFC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module):
    def __init__(self, input_size, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(input_size, 256)
        self.act1 = torch.sin
        self.fc2 = ConcatFC(256, 256)
        self.act2 = torch.sin
        self.fc3 = ConcatFC(256, input_size)
        self.act3 = torch.sin
        self.nfe = 0
        self.f_coeffi = -1

    def forward(self, t, x):
        self.nfe += 1
        out = self.f_coeffi*self.fc1(t, x)
        out = self.act1(out)
        out = self.f_coeffi*self.fc2(t, out)
        out = self.act2(out)
        out = self.f_coeffi*self.fc3(t, out)
        out = self.act3(out)    
        return out    
    
class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.endtime = 5
        self.tol = 1e-3
        self.integration_time = torch.tensor([0, self.endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODE_wrapped_model(nn.Module):
    def __init__(self, model):
        super(ODE_wrapped_model, self).__init__()
        if hasattr(model, 'fc'):
            classes, layer_shape = model.fc.weight.shape
            self.linear = copy.deepcopy(model.fc)
        elif hasattr(model, 'logits'):
            classes, layer_shape = model.logits.weight.shape
            self.linear = copy.deepcopy(model.logits)
        else:
            classes, layer_shape = model.linear.weight.shape
            self.linear = copy.deepcopy(model.linear)

        self.odefunc = ODEfunc_mlp(layer_shape, 0)
        self.feature_layers = ODEBlock(self.odefunc) 
        self.backbone = FeatureExtractor(model)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.feature_layers(x)
        x = self.linear(x)
        return x
    
    def df_dz_regularizer(self, f, z):
        regu_diag = 0.
        regu_offdiag = 0.0
        time_df = 1
        exponent = 1.0
        exponent_off = 0.1
        trans = 1.0
        transoffdig = 1.0
        numm = 16 
        for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
            ode_lambda_fn = lambda x: self.odefunc(torch.tensor(time_df).cuda(), x)
            batchijacobian = torch.autograd.functional.jacobian(ode_lambda_fn, z[ii:ii+1,...], create_graph=True)
            batchijacobian = batchijacobian.view(z.shape[1],-1)
            if batchijacobian.shape[0]!=batchijacobian.shape[1]:
                raise Exception("wrong dim in jacobian")
            
            tempdiag = torch.diagonal(batchijacobian, 0)
            regu_diag += torch.exp(exponent*(tempdiag+trans))
            offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).cuda()+0.5)*2), dim=0)
            off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
            regu_offdiag += off_diagtemp

        print('diag mean: ',tempdiag.mean().item())
        print('offdiag mean: ',offdiat.mean().item())
        return regu_diag/numm, regu_offdiag/numm
    
    def f_regularizer(self, f, z):
        time_df = 1
        exponent_f = 50
        tempf = torch.abs(self.odefunc(torch.tensor(time_df).cuda(), z))
        regu_f = torch.pow(exponent_f*tempf,2)
        print('tempf: ', tempf.mean().item())
        return regu_f
    
    def ode_training(self):
        weight_diag = 10
        weight_offdiag = 0
        weight_f = 0.1
        print('==> Prepairing data..')
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)
        
        optimizer = torch.optim.Adam(self.feature_layers.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
    
        for epoch in range(0):
            for i, (images, labels) in enumerate(trainloader):
                optimizer.zero_grad()
                images, labels = images.cuda(), labels.cuda()
                y00=self.backbone(images)

                regu1, regu2  = self.df_dz_regularizer(self.odefunc, y00)
                regu1, regu2 = regu1.mean(), regu2.mean()
                regu3 = self.f_regularizer(self.odefunc, y00)
                regu3 = regu3.mean()
                loss = weight_f*regu3 + weight_diag*regu1+ weight_offdiag*regu2
                loss.backward()
                optimizer.step()
        
        optimizer = torch.optim.Adam([{'params': self.odefunc.parameters(), 'lr': 1e-5, 'eps':1e-6,},
                            {'params': self.linear.parameters(), 'lr': 1e-2, 'eps':1e-4,}], amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            images, labels = images.cuda(), labels.cuda()
            outputs = self.forward(images)
            loss = criterion(outputs, labels)
            print("Classification loss", loss)
            loss.backward()
            optimizer.step()
        return 
