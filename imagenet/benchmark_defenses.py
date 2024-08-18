from robustbench.utils import load_model
from defenses.riuf import OurModel
from robustbench.eval import benchmark


for model_name in ['Salman2020Do_50_2']:
    print(model_name)
    model = load_model(model_name=model_name, dataset='imagenet',threat_model='Linf').cuda()
    model.eval()
    print(model)
    clean_acc,robust_acc=benchmark(model,n_examples=5000,dataset='imagenet',data_dir='data',preprocessing='Res256Crop224',threat_model='Linf', eps=0.0156862745)
    print("clean: ",clean_acc,"robust: ",robust_acc)

    #our_model = OurModel(model, 1000).cuda()
    #our_model.eval()
    #clean_acc,robust_acc=benchmark(our_model,n_examples=5000,dataset='imagenet',data_dir='data',preprocessing='Res256Crop224',threat_model='Linf', eps=0.0156862745)
    #print("clean: ",clean_acc,"robust: ",robust_acc)
