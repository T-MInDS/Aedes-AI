import matplotlib.pyplot as plt

def plot_loss(history, model):
    plt.plot(history.history['loss'], c='k', linestyle='-')
    plt.plot(history.history['val_loss'], c='k', linestyle='--')
    plt.title(f'{model} Loss with Subsequent Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'figures/training/{model}_loss.png')

def plot_r2(history, model):
    plt.plot(history.history['r2_keras'], c='k', linestyle='-')
    plt.plot(history.history['val_r2_keras'], c='k', linestyle='--')
    plt.title(f'{model} $R^{2}$ with Subsequent Epochs')
    plt.ylabel('$R^{2}$')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'figures/training/{model}_r2.png')