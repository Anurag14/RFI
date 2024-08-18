import copy
from robustbench.data import load_cifar100, load_cifar10
from robustbench.utils import load_model
from autoattack import AutoAttack
from defenses.riuf import OurModel
from defenses.sodef import ODE_wrapped_model 
from defenses.anti_adv import anti_adversary_wrapper 

def benchmark(model,x_test, y_test, attack):
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=[attack])
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=500)
    return 

#model_name_cifar10 = 'Engstrom2019Robustness'
#model_name_cifar100 = 'Addepalli2022Efficient_RN18'
#model_name_cifar10 = 'Carmon2019Unlabeled'
#model_name_cifar100 = 'Pang2022Robustness_WRN28_10'
#model_name_cifar10 = 'Wang2023Better_WRN-28-10'
#model_name_cifar100 = 'Wang2023Better_WRN-28-10'
model_name_cifar10 = 'Rice2020Overfitting'
#model_name_cifar100 = 'Rice2020Overfitting'
attacks_to_run = ['fab-t', 'square']

x_test, y_test = load_cifar10(n_examples=10000)
baseline = load_model(model_name=model_name_cifar10, dataset='cifar10', threat_model='Linf').cuda()
our_model = OurModel(copy.deepcopy(baseline), 10).cuda()
antiadv = anti_adversary_wrapper(copy.deepcopy(baseline)).cuda()
sodef = ODE_wrapped_model(copy.deepcopy(baseline)).cuda()
print(model_name_cifar10)
for attack in attacks_to_run:
    print("ATTACK: ", attack)    
    benchmark(baseline, x_test, y_test, attack)
    print("Benchmark for baseline is complete")
    print("===============================================================")
    benchmark(our_model, x_test, y_test, attack)
    print("Benchmark for our model is complete")
    print("===============================================================")
    benchmark(antiadv, x_test, y_test, attack)
    print("Benchmark for antiadv is complete")
    print("===============================================================")
    benchmark(sodef, x_test, y_test, attack)
    print("Benchmark for SODEF is complete")
"""
x_test, y_test = load_cifar100(n_examples=10000)
baseline = load_model(model_name=model_name_cifar100, dataset='cifar100', threat_model='Linf').cuda()
our_model = OurModel(copy.deepcopy(baseline), 100).cuda()
antiadv = anti_adversary_wrapper(copy.deepcopy(baseline)).cuda()
sodef = ODE_wrapped_model(copy.deepcopy(baseline)).cuda()
print(model_name_cifar100)
for attack in attacks_to_run:
    print("ATTACK: ", attack)
    benchmark(baseline, x_test, y_test, attack)
    print("Benchmark for baseline is complete")
    print("===============================================================")
    benchmark(our_model, x_test, y_test, attack)
    print("Benchmark for our model is complete")
    print("===============================================================")
    benchmark(antiadv, x_test, y_test, attack)
    print("Benchmark for antiadv is complete")
    print("===============================================================")
    benchmark(sodef, x_test, y_test, attack)
    print("Benchmark for SODEF complete")
"""
