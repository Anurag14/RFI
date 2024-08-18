
import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from autoattack import AutoAttack
from robustbench.eval import benchmark
from defenses.riuf import OurModel
from defenses.sodef import ODE_wrapped_model 
from defenses.anti_adv import anti_adversary_wrapper 
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
## iterate through the cifar 10 dataset to obtain covariance 
x_test, y_test = load_cifar10(n_examples=10000)

def auto_benchmark(model, x_test, y_test):
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    return 
device = torch.device("cuda")
for model_name in ['Wang2023Better_WRN-28-10']:
    print(model_name)
    model = load_model(model_name=model_name, dataset='cifar100', threat_model='Linf').cuda()
    model.eval()
    clean_acc, robust_acc = benchmark(model,dataset='cifar100', batch_size=250, threat_model='Linf', eps=8/255)
    our_model = OurModel(model, 100, hard=True)
    print(our_model)
    print("Sota model altered: Now running new benchmark...")
    #clean_acc, robust_acc = benchmark(our_model,dataset='cifar100', batch_size=250, threat_model='Linf', eps=8/255, device=device)
    #clean_acc, robust_acc = benchmark(model,dataset='cifar100', batch_size=250, threat_model='Linf', eps=8/255, device=device)
