import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=111550159):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss(epochs, train_losses, valid_losses, save_name='loss_curve.png'):
    Epochs = range(epochs)
    plt.figure(figsize=(10, 6))
    plt.plot(Epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(Epochs, valid_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
