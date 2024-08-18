import torch
import matplotlib.pyplot as plt
checkpoints = ['basic_training', 'pgd_adversarial_training', 'interpolated_adversarial_training', 'basic_training_with_robust_dataset']
#eigenvalues = []
for checkpoint in checkpoints:
    value = torch.load(checkpoint+'_values.pt')
    #eigenvalues.append(value.numpy())
    plt.plot(value.numpy()[:50], label=checkpoint)
plt.ylabel("Eigenvalues of Feature covariance")
plt.legend()
plt.savefig('plot.png')
