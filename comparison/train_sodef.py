from robustbench.data import load_cifar10
from robustbench.utils import load_model
from autoattack import AutoAttack
from defenses.riuf import OurModel
from defenses.sodef import ODE_wrapped_model 
from defenses.anti_adv import anti_adversary_wrapper 
## iterate through the cifar 10 dataset to obtain covariance 
x_test, y_test = load_cifar10(n_examples=10000)

def benchmark(model, x_test, y_test):
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    return

model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
model.cuda()
model.eval()
our_model = ODE_wrapped_model(model)
our_model.cuda()
print("training starts")
our_model.ode_training()
our_model.eval()
benchmark(our_model, x_test, y_test)
