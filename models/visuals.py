import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'], c='k', linestyle='-')
    plt.plot(history.history['val_loss'], c='k', linestyle='--')
    plt.title('GRU Model Loss with Subsequent Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_r2(history):
    plt.plot(history.history['r2_keras'], c='k', linestyle='-')
    plt.plot(history.history['val_r2_keras'], c='k', linestyle='--')
    plt.title('GRU Model $R^{2}$ with Subsequent Epochs')
    plt.ylabel('$R^{2}$')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()