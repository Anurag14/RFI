class Config:
    def __init__(self, dataset_name=None, model_name='Resnet18', epochs=100, attack_name='FGSM', adv_training=False):
        self.model_name = model_name
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.attack_name = attack_name
        self.adv_training = adv_training
